from __future__ import annotations

from discord.ext import commands
import discord

import json
import asyncio

import numpy as np
import cv2 
from enum import Enum, auto
from copy import copy
from functools import cached_property
from typing import Iterator, BinaryIO, Optional
from PIL import Image

from io import BytesIO

import settings
from pathlib import Path

import logging
_log = logging.getLogger(__name__)


class Pieces(Enum):
	PAWN = auto()
	KNIGHT = auto()
	BISHOP = auto()
	ROOK = auto()
	QUEEN = auto()
	KING = auto()

class FenPieces(Enum):
	P =  Pieces.PAWN;	p = Pieces.PAWN
	N =  Pieces.KNIGHT;	n = Pieces.KNIGHT
	B =  Pieces.BISHOP;	b = Pieces.BISHOP
	R =  Pieces.ROOK;	r = Pieces.ROOK
	Q =  Pieces.QUEEN;	q = Pieces.QUEEN
	K =  Pieces.KING;	k = Pieces.KING

KNIGHT_MOVES = [( 1, -2), ( 2, -1), ( 2,  1), ( 1,  2), (-1,  2), (-2,  1), (-2, -1), (-1, -2),]

def checkmove(f):
	def wrapper(self, *args):
		move = self.fromalpha(args[0]) 
		if move in self._cached:
			for i in self._cached[move]:
				yield i
		else:
			hist = []
			self._cached[move] = []
			for i in f(self, *args):
				a = self.toalpha(i)
				if a not in hist: hist.append(a)
				else: continue
				try:
					if not self._protomove(args[0], i).ischeck:
						self._cached[move].append(i)
						yield i
				except ValueError as e:
					continue
	return wrapper

def pawn_(self, pos):
	turn = (lambda x,y: self.board[y,x] > 8)
	piece = self.board[pos[::-1]]
	start = 6
	dir_ = -1
	if piece > 8:
		start = 1; dir_=1

	if self.board[pos[1] + dir_, pos[0]] == 0:
		yield (pos[0], pos[1] + dir_)

	m = (pos[0], pos[1] + dir_ * 2)
	if start == pos[1] and self.board[m[1], m[0]] == 0:
		yield m

	ld = pos[0] - 1, pos[1] + dir_ 
	rd = pos[0] + 1, pos[1] + dir_ 
	if self.board[ld[::-1]] and turn(*ld) != turn(*pos):
		yield ld
	if self.board[rd[::-1]] and turn(*rd) != turn(*pos):
		yield rd

	if self.en_passant != '-':
		en_passant = self.fromalpha(self.en_passant) 
		if turn(*pos) and pos[1] == 4:
			if abs(en_passant[0] - pos[0]) == 1:
				yield (en_passant[0], pos[1] + 1)
		if not turn(*pos) and pos[1] == 3:
			if abs(en_passant[0] - pos[0]) == 1:
				yield (en_passant[0], pos[1] - 1)

def bishop_(self, pos):
	turn = (lambda x,y: self.board[y,x] > 8)
	for d in range(1, 8):
		x, y = pos[0] + d, pos[1] + d
		if not (0 <= x < 8): break
		if not (0 <= y < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)

	for d in range(1, 8):
		x, y = pos[0] + d, pos[1] - d
		if not (0 <= x < 8): break
		if not (0 <= y < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)

	for d in range(1, 8):
		x, y = pos[0] - d, pos[1] - d
		if not (0 <= x < 8): break
		if not (0 <= y < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)

	for d in range(1, 8):
		x, y = pos[0] - d, pos[1] + d
		if not (0 <= x < 8): break
		if not (0 <= y < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)

def knight_(self, pos):
	turn = (lambda x,y: self.board[y,x] > 8)
	for dx, dy in KNIGHT_MOVES:
		x, y = pos[0] + dx, pos[1] + dy
		if not (0 <= x < 8): continue
		if not (0 <= y < 8): continue
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): continue
			yield (x, y)
			continue
		yield (x, y)

def rook_(self, pos):
	turn = (lambda x,y: self.board[y,x] > 8)
	for d in range(1, 8):
		x, y = pos[0] - d, pos[1]
		if not (0 <= x < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)

	for d in range(1, 8):
		x, y = pos[0] + d, pos[1]
		if not (0 <= x < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)

	for d in range(1, 8):
		x, y = pos[0], pos[1] - d
		if not (0 <= y < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)

	for d in range(1, 8):
		x, y = pos[0], pos[1] + d
		if not (0 <= y < 8): break
		if self.board[y, x] != 0:
			if turn(*pos) == turn(x, y): break
			yield (x, y)
			break
		yield (x, y)


def king_(self, pos):
	turn = (lambda x,y: self.board[y,x] > 8)
	for dx in [-1, 0, 1]:
		for dy in [-1, 0, 1]:
			x, y = pos[0] + dx, pos[1] + dy
			if not (0 <= x < 8): continue
			if not (0 <= y < 8): continue
			if dx == dy == 0: continue
			if (self.board[y, x]) != 0:
				if turn(*pos) == turn(x, y): continue
				yield (x, y)
			yield (x, y)
	if self.turn == 'w':
		if self.Kcastle:
			for p in ((5, 0), (6, 0)):
				if self.board[p[::-1]]: break
				if self._protomove(pos, p).ischeck: break
			else:
				yield (6, 0)
		if self.Qcastle:
			for p in ((3, 0), (2, 0) ):
				if self.board[p[::-1]]: break
				if self._protomove(pos, p).ischeck: break
			else:
				yield (2, 0)
	if self.turn == 'b':
		if self.Kcastle:
			for p in ((5, 7), (6, 7)):
				if self.board[p[::-1]]: break
				if self._protomove(pos, p).ischeck: break
			else:
				yield (6, 7)
		if self.Qcastle:
			for p in ((3, 7), (2, 7)):
				if self.board[p[::-1]]: break
				if self._protomove(pos, p).ischeck: break
			else:
				yield (2, 7)

def queen_(self, pos):
	for i in rook_(self, pos):
		yield i
	for i in bishop_(self, pos):
		yield i


class Engine:
	def __init__(self, board, turn, castling, en_passant, half, full):
		self.board = board
		self.turn = turn
		self.castling = castling
		self.en_passant = en_passant
		self.half = half
		self.full = full

		self._check = False
		self._cached = {}
	
	@property
	def Kcastle(self) -> bool: return self.castling & 0b1000
	@property
	def Qcastle(self) -> bool: return self.castling & 0b0100
	@property
	def kcastle(self) -> bool: return self.castling & 0b0010
	@property
	def qcastle(self) -> bool: return self.castling & 0b0001

	def tofen(self) -> str:
		w = ''
		for y in self.board[::-1]:
			i = 0
			for x in y:
				if x == 0:
					i += 1
					continue
				if (i != 0):
					w += str(i);
					i = 0
				cap = False
				if x > 8:
					x -= 1 << 3; cap=True
				piece = FenPieces(Pieces(x))
				w += piece.name if cap else piece.name.lower()
			if i != 0: w += str(i)
			w += '/'
		w += f' {self.turn} '

		w += ('K' if self.castling & 0b1000 else '') +\
		     ('Q' if self.castling & 0b0100 else '') +\
		     ('k' if self.castling & 0b0010 else '') +\
		     ('q' if self.castling & 0b0001 else '')

		w += f' {self.en_passant} {self.half} {self.full}'
		return w

	def copy(self) -> 'Engine':
		return Engine(self.board.copy(),
				copy(self.turn),
				copy(self.castling),
				copy(self.en_passant),
				copy(self.half),
				copy(self.full))

	@cached_property
	def turn_king(self) -> tuple:
		v = Pieces.KING.value
		if self.turn == 'w':	v += 1 << 3
		r = np.stack(np.where(self.board == v), axis=1)
		if not r.size > 0:
			raise ValueError('no king')
		return tuple(r[0][::-1])

	@cached_property
	def ischeck(self) -> bool:
		turn = (lambda x,y: self.board[y,x] > 8)
		king = self.turn_king
		for x,y in bishop_(self, king):
			if self.board[y, x] != 0 and turn(x, y) != turn(*king):
				p = self.board[y, x]
				if p > 8: p -= 1 << 3;
				if p in (Pieces.BISHOP.value, Pieces.QUEEN.value):
					return True
		for x,y in rook_(self, king):
			if self.board[y, x] != 0 and turn(x, y) != turn(*king):
				p = self.board[y, x]
				if p > 8: p -= 1 << 3;
				if p in (Pieces.ROOK.value, Pieces.QUEEN.value):
					return True
		for x,y in knight_(self, king):
			if self.board[y, x] != 0 and turn(x, y) != turn(*king):
				p = self.board[y, x]
				if p > 8: p -= 1 << 3;
				if p in (Pieces.KNIGHT.value,):
					return True
		dir_ = -1;
		if turn(*king): dir_ = 1;
		for x,y in [(king[0]-1, king[1]+dir_), (king[0]+1,king[1]+dir_)]:
			if self.board[y, x] != 0 and turn(x, y) != turn(*king):
				p = self.board[y, x]
				if p > 8: p -= 1 << 3;
				if p in (Pieces.PAWN.value,):
					return True
		return False

	@cached_property
	def ischeckmate(self) -> bool:
		king = self.turn_king
		if self.ischeck and len([*king_(self, self.board, king)]) == 0:
			return True
		return False

	def _protomove(self, src, dst, final=False, promote: Optional[Pieces]=None):
		"""DO NOT USE

		RESERVED FOR IMPLEMENTATION PURPOSES, COULD BE DANGEROUS"""
		if isinstance(src, str): src = self.fromalpha(src) 
		if isinstance(dst, str): dst = self.fromalpha(dst) 

		src = tuple(src)
		if self.board[src[::-1]] == 0: raise ValueError('no pieces selected')
		if self.board[src[::-1]] > 8 and self.turn == 'b': raise ValueError('black turn')
		if self.board[src[::-1]] < 8 and self.turn == 'w': raise ValueError('white turn')
		if self.board[dst[::-1]] != 0 and ((self.board[src[::-1]] < 8) == (self.board[dst[::-1]] < 8)):
			raise ValueError('capturing own piece', src, dst)

		new = self.copy()
		new.board[dst[::-1]], new.board[src[::-1]] = self.board[src[::-1]], 0

		if final:
			if self.turn == 'w' and self.board[src[::-1]] == Pieces.KING.value + (1 << 3) and self.Kcastle and dst[0] == 6 and src[1] == dst[1] == 0:
				new.board[0, 5], new.board[0, 7] = Pieces.ROOK.value + (1 << 3), 0
			if self.turn == 'w' and self.board[src[::-1]] == Pieces.KING.value + (1 << 3) and self.Qcastle and dst[0] == 2 and src[1] == dst[1] == 0:
				new.board[0, 3], new.board[0, 0] = Pieces.ROOK.value + (1 << 3), 0
			if self.turn == 'b' and self.board[src[::-1]] == Pieces.KING.value            and self.kcastle and dst[0] == 6 and src[1] == dst[1] == 7:
				new.board[7, 5], new.board[7, 7] = Pieces.ROOK.value, 0
			if self.turn == 'b' and self.board[src[::-1]] == Pieces.KING.value            and self.qcastle and dst[0] == 2 and src[1] == dst[1] == 7:
				new.board[7, 3], new.board[7, 0] = Pieces.ROOK.value, 0
			if promote:
				if self.turn == 'w' and self.board[src[::-1]] == Pieces.PAWN.value + (1 << 3) and dst[1] == 7:
					new.board[dst[::-1]] = promote.value + (1 << 3) 
				if self.turn == 'b' and self.board[src[::-1]] == Pieces.PAWN.value            and dst[1] == 0:
					new.board[dst[::-1]] = promote.value + (1 << 3) 
			# TODO en passant final 
			if self.en_passant != '-':
				en_passant = self.fromalpha(self.en_passant)
				if src[1] == en_passant[1] and np.abs(en_passant[0] - src[0]) == 1:
					new.board[en_passant[::-1]] = 0	

		return new

	def move(self, src: str | tuple | list, dst: str | tuple | list, promote: Optional[Pieces]=None) -> 'Engine':
		if isinstance(src, str): src = self.fromalpha(src) 
		if isinstance(dst, str): dst = self.fromalpha(dst) 

		assert dst in self.moves(src), 'invalid move'
		new = self._protomove(src, dst, final=True, promote=promote)
		new.turn = 'wb'[self.turn == 'w']
		new.en_passant = '-'

		piece = self.board[src[::-1]]
		if piece > 8: piece -= 1 << 3
		piece = Pieces(piece)
		if self.turn == 'w' and piece == Pieces.PAWN and abs(src[1] - dst[1]) == 2:
			new.en_passant = self.toalpha((src[0], 3))
		if self.turn == 'b' and piece == Pieces.PAWN and abs(src[1] - dst[1]) == 2:
			new.en_passant = self.toalpha((src[0], 4))

		new.half += 1
		new.full += 1
		if (self.board[dst[::-1]]) != 0:
			new.half = 0
		if self.turn == 'w' and (new.ischeck or (piece == Pieces.ROOK and src[0] == 7)):
			new.castling &= 0b0111
		if self.turn == 'w' and (new.ischeck or (piece == Pieces.ROOK and src[0] == 0)):
			new.castling &= 0b1011
		if self.turn == 'b' and (new.ischeck or (piece == Pieces.ROOK and src[0] == 7)):
			new.castling &= 0b1101
		if self.turn == 'b' and (new.ischeck or (piece == Pieces.ROOK and src[0] == 0)):
			new.castling &= 0b1110
		return new

	@checkmove
	def moves(self, square: str | tuple | list) -> Iterator[tuple[int, int]]:
		pos = self.fromalpha(square)

		piece = self.board[pos[1], pos[0]]
		assert piece != 0, 'empty space'
		assert (piece > 8) == (self.turn == 'w'), 'invalid turn'

		dir_ = -1
		if piece > 8:
			piece -= 1 << 3; dir_ = 1

		if Pieces(piece) == Pieces.PAWN:
			for i in pawn_(self, pos):
				yield i
		elif Pieces(piece) == Pieces.KNIGHT:
			for i in knight_(self, pos):
				yield i
		elif Pieces(piece) == Pieces.BISHOP:
			for i in bishop_(self, pos):
				yield i
		elif Pieces(piece) == Pieces.ROOK:
			for i in rook_(self, pos):
				yield i
		elif Pieces(piece) == Pieces.KING:
			for i in king_(self, pos):
				yield i
		elif Pieces(piece) == Pieces.QUEEN:
			for i in queen_(self, pos):
				yield i
		else:
			raise ValueError('invalid piece')

	@classmethod
	def fromfen(cls, fen: str) -> 'Engine':
		fields = fen.split(' ')
		assert len(fields) == 6, 'invalid fen layout'

		placement = fields[0]
		board = np.zeros((8, 8), dtype='u1')
		pos = [0, 7]

		for i,c in enumerate(placement):
			if c in '12345678':
				pos[0] += int(c)
				assert pos[0] <= 8, f'invalid fen board {i}: {c}'
			if c in 'pnbrqkPNBRQK':
				value = FenPieces[c].value.value 
				if c.isupper():
					value += 1 << 3 # if upper white
				board[pos[1], pos[0]] = value
				pos[0] += 1
				assert pos[0] <= 8, f'invalid fen board{i}: {c}'
			if c == '/':
				pos[1] -= 1; pos[0] = 0

		turn = fields[1]
		assert turn in 'wb', 'invalid fen turn'

		assert len(fields[2]) <= 4 and all(map(lambda x: x in 'KkQq', fields[2])), 'invalid fen castle'
		castling = (('K' in fields[2]) << 3) + (('Q' in fields[2]) << 2) + (('k' in fields[2]) << 1) + (('q' in fields[2]) << 0)

		en_passant = fields[3]
		assert en_passant == '-' or (en_passant[0] in 'abcdefgh' and en_passant[1] in '12345678'), 'invalid fen en passant'

		half = fields[4]
		assert half.isnumeric(), 'invalid fen half move clock'
		half = int(half)

		full = fields[5]
		assert full.isnumeric(), 'invalid fen full move number'
		full = int(full)

		return cls(board, turn, castling, en_passant, half, full)	

	def __repr__(self):
		w = ''
		for y in self.board[::-1]:
			for x in y:
				if x == 0:
					w += '  '; continue
				cap = False
				if x > 8:
					x -= 1 << 3; cap=True
				piece = FenPieces(Pieces(x))
				w += piece.name if cap else piece.name.lower()
				w += ' '
			w += '\n'
		return w[:-2]
	
	@classmethod
	def fromalpha(cls, square: str | tuple | list) -> tuple:
		assert isinstance(square, (str, tuple, list)), 'invalid type'

		assert bool(square), 'empty move' 
		assert len(square) == 2, 'invalid move'
		if isinstance(square, (tuple, list)):
			return copy(square)
		assert square[0] in 'abcdefgh' and square[1] in '12345678', 'square doesn\'t exist'
		return 'abcdefgh'.index(square[0]), int(square[1]) - 1

	@classmethod
	def toalpha(cls, pos: tuple | list):
		assert isinstance(pos, (tuple, list))
		return f'{chr(97+pos[0])}{str(pos[1] + 1)}'

def save_json(path, obj):
	with open(path, 'w', encoding='utf8') as f:
		json.dump(obj, f, indent=4)

def load_json(path):
	with open(path, 'r', encoding='utf8') as f:
		data = json.load(f)
	return data

def loadimg(path, size=None):
	if size is not None:
		return Image.open(path).resize(size).convert('RGBA')
	return Image.open(path).convert('RGBA')

def circular_grad(size, inner, outer, factor=1):
	d = np.power(np.abs(np.linspace(-1, 1, size)), factor)
	d[:size // 2] *= -1
	xx, yy = np.meshgrid(d, d)

	dist = np.sqrt(xx ** 2 + yy ** 2)
	inner = np.array(inner)[None, None, :] 
	outer = np.array(outer)[None, None, :] 

	dist = (dist / np.sqrt(2))[:, :, None]
	img = dist * outer + (1 - dist) * inner
	img = (img * 255).astype('u1')
	r = Image.fromarray(img, mode='RGBA')
	return r

def rot90(v):
	return np.array([v[1], -v[0]])

def knight_arrow(img, start, end, thickness):
	d = np.abs(start - end)
	cv2args = [(127, 127, 127), thickness, 16, 0]
	if d[0] > d[1]:
		cv2.line(img,
			[start[0], start[1]],
			[end[0], start[1]],
			*cv2args[:-1]
		)
		start[0] = end[0]
	else:
		cv2.line(img,
			[start[0], start[1]],
			[start[0], end[1]],
			*cv2args[:-1]
		)
		start[1] = end[1]

	dir_ = end.astype(int) - start.astype(int)
	triangle_len = triangle(img, end, dir_, thickness, angle=settings.chess.angle, unit=settings.chess.unit)

	end = (end - triangle_len * (dir_ / np.sqrt(np.sum(np.power(dir_,2))))).astype('u8')
	cv2.line(img, start, end, *cv2args)



def trlen(t, s):
	return np.sqrt(s * t ** 2 - t ** 2)

def cot(a, *, unit='deg'):
	if unit == 'deg':
		return 1 / np.tan(a * np.pi / 180)
	if unit == 'rad':
		return 1 / np.tan(a)
	raise ValueError(f'invalid unit {unit}')

def getsquish(t, a, unit='deg'):
	if unit == 'deg':
		a = a * np.pi / 180
	elif unit != 'rad':
		raise ValueError(f'invalid unit {unit}')
	return (np.tan(a) ** 2 + 1) * (cot(a, unit='rad') ** 2)

def triangle(img, pos, dir_, thickness, *, squish=None, angle=None, unit='deg'):
	"""squish: 1 = line, 2 = right angle"""
	assert (squish is not None or angle is not None) and not (squish is not None and angle is not None), 'Only one of squish and angle can be specified'
	t = thickness * 1.5
	dir_ = dir_ / np.sqrt(np.sum(np.power(dir_,2))) 
	d = rot90(dir_)

	if angle:
		squish = getsquish(t, angle / 2, unit=unit)
	pc = trlen(t, squish)

	a = np.array(pos + d * t, dtype=int) - pc * dir_
	b = np.array(pos - d * t, dtype=int) - pc * dir_
	c = pos + dir_ * pc                  - pc * dir_

	pts = np.array([a, b, c], dtype=np.int32).reshape((-1, 1, 2))

	cv2.fillPoly(img, [pts], color=(127, 127, 127))
	return pc

def rectangle(img, pos, cell_size, color):
	mask = np.zeros(np.array(img).shape, dtype=np.uint8)

	start = np.array(pos, dtype=np.int32) * cell_size
	end = start + cell_size

	cv2.rectangle(mask, start, end, (127, 127, 127), -1)

	mask[mask[::, ::, 1] == 127] = color
	temp = Image.fromarray(mask)
	img.paste(temp, [0, 0], temp)
	return temp

def arrow(img, start, end, thickness, cell_size):
	c = cell_size // 2
	startpos = np.array(start, dtype='u8')
	endpos = np.array(end, dtype='u8')

	start = startpos * cell_size + cell_size // 2
	end = endpos * cell_size + cell_size // 2

	mask = np.zeros(np.array(img).shape, dtype=np.uint8)
	cv2args = [(127, 127, 127), thickness, 16, 0]

	if np.all(startpos.astype(int) - endpos.astype(int) != 0) and np.sum(np.abs(startpos.astype(int) - endpos.astype(int))) == 3:
		knight_arrow(mask, start, end, thickness)
	else:
		dir_ = end.astype(int) - start.astype(int)
		triangle_len = triangle(mask, end, dir_, thickness, angle=settings.chess.angle, unit=settings.chess.unit)

		end = (end - triangle_len * (dir_ / np.sqrt(np.sum(np.power(dir_,2))))).astype('u8')

		cv2.line(mask, start, end, *cv2args)

	mask[mask[::, ::, 1] == 127] = (127, 127, 127, 180)

	arrow_ = Image.fromarray(mask)
	img.paste(
			arrow_,
			[0, 0],
			arrow_
			)

	return arrow_

class Images:
	WIDTH, HEIGHT = 1600, 1600
	board	= loadimg(settings.chess.img / '1600.png', (WIDTH, HEIGHT))

	_f = np.array(loadimg(settings.chess.img / 'font.png'))
	_f[_f[::, ::, 1] > 238] = 0
	font = Image.fromarray(_f)
	del _f

	psize	= 150

	bp		= loadimg(settings.chess.img / 'bp.png', (psize, psize))
	bn		= loadimg(settings.chess.img / 'bn.png', (psize, psize))
	bb		= loadimg(settings.chess.img / 'bb.png', (psize, psize))
	br		= loadimg(settings.chess.img / 'br.png', (psize, psize))
	bq		= loadimg(settings.chess.img / 'bq.png', (psize, psize))
	bk		= loadimg(settings.chess.img / 'bk.png', (psize, psize))
	

	wp		= loadimg(settings.chess.img / 'wp.png', (psize, psize))
	wn		= loadimg(settings.chess.img / 'wn.png', (psize, psize))
	wb		= loadimg(settings.chess.img / 'wb.png', (psize, psize))
	wr		= loadimg(settings.chess.img / 'wr.png', (psize, psize))
	wq		= loadimg(settings.chess.img / 'wq.png', (psize, psize))
	wk		= loadimg(settings.chess.img / 'wk.png', (psize, psize))

	U=1<<3
	pieces	= {
			Pieces.PAWN.value:		bp,
			Pieces.KNIGHT.value:	bn,
			Pieces.BISHOP.value:	bb,
			Pieces.ROOK.value:		br,
			Pieces.QUEEN.value:		bq,
			Pieces.KING.value:		bk,

			Pieces.PAWN.value+U:	wp,
			Pieces.KNIGHT.value+U:	wn,
			Pieces.BISHOP.value+U:	wb,
			Pieces.ROOK.value+U:	wr,
			Pieces.QUEEN.value+U:	wq,
			Pieces.KING.value+U:	wk
	}

	
	rad = circular_grad(200, [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], factor=1 / 2**.5)

	_f = np.zeros([psize, psize, 4], dtype=np.uint8)
	cv2.circle(_f, [psize // 2, psize // 2], 30, [127, 127, 127], -1) 
	_f[_f[::, ::, 1] == 127] = (127, 187, 187, 130)
	possible = Image.fromarray(_f)
	del _f

def generate_img(*, images: Images, game: Engine=None, fen: str=None, fp: BinaryIO=None,
			 lastmove: str=None, selected: str=None) -> Image:
	"""Creates a PIL Image from a chess board (Engine object or FEN string)."""
	assert not (lastmove is not None and selected is not None), "Only one of selected and lastmove allowed"
	if game is not None and fen is not None:
		raise ValueError('Only one source can be provided')
	if game:
		assert isinstance(game, Engine)
	if fen:
		game = Engine.fromfen(fen)

	board = images.board.copy()
	xstep = images.WIDTH // 8
	ystep = images.HEIGHT // 8
	xpad = (xstep - images.psize) // 2
	ypad = (ystep - images.psize) // 2

	tk = game.turn_king
	tk = [tk[0], 7 - tk[1]]

	if selected:
		if isinstance(selected, str): src = game.fromalpha(selected) 
		src = [src[0], 7-src[1]]
		rectangle(board, src, 200, (255, 167, 167, 200))


	for y,col in enumerate(game.board[::-1]):
		for x,piece in enumerate(col):
			if game.ischeck and tk[0] == x and tk[1] == y:
				board.paste(
						images.rad,
						(xstep * x, ystep * y),
						images.rad
					)
			if piece:
				board.paste(
						images.pieces[piece],
						(xstep * x + xpad, ystep * y + ypad),
						images.pieces[piece]
					)
	for n in range(8):
		x = n * 39
		if n % 2 == 0:
			x += 624
		c = images.font.crop([x, 0, x+39, 60])
		board.paste(
				c,
				(10, images.HEIGHT - ystep * (n + 1) + 10),
				c
			)

		x = 312 + n * 39
		if n % 2 == 0:
			x += 624
		c = images.font.crop([x, 0, x+39, 60])
		board.paste(
				c,
				(xstep * n + xstep - 49, images.HEIGHT - 70),
				c
			)

	if selected:
		if isinstance(selected, str): src = game.fromalpha(selected) 
		src = [src[0], 7-src[1]]
		for x,y in game.moves(selected):
			y = 7-y
			board.paste(
					images.possible,
					(xstep * x + xpad, ystep * y + ypad),
					images.possible
				)

	if lastmove:
		src = lastmove[:2]; dst = lastmove[2:]
		if isinstance(src, str): src = game.fromalpha(src) 
		if isinstance(dst, str): dst = game.fromalpha(dst) 
		src = [src[0], 7-src[1]]
		dst = [dst[0], 7-dst[1]]

		arrow(board, src, dst, settings.chess.thickness, 200)
	if fp:
		board.save(fp, 'PNG')
	return board

class _userselect(discord.ui.UserSelect):
	def __init__(self, user):
		self.chess_user = user
		self.future = asyncio.Future()
		super().__init__(placeholder="Choose an opponent", min_values=1, max_values=1)
	
	async def callback(self, interaction: discord.Interaction):
		if interaction.user == self.chess_user:
			self.future.set_result((self.values[0], interaction.response))

class _promoteselect(discord.ui.Select):
	def __init__(self, user):
		self.chess_user = user
		self.future = asyncio.Future()
		options = [
				discord.SelectOption(label='bishop', description='bishop', value=Pieces.BISHOP.value),
				discord.SelectOption(label='knight', description='knight', value=Pieces.KNIGHT.value),
				discord.SelectOption(label='rook',   description='rook',   value=Pieces.ROOK.value),
				discord.SelectOption(label='queen',  description='queen',  value=Pieces.QUEEN.value),
			]
		super().__init__(placeholder='Choose what to promote to', min_values=1, max_values=1, options=options)
	async def callback(self, interaction: discord.Interaction):
		if interaction.user == self.chess_user:
			self.future.set_result((self.values[0], interaction.response))

class Chess(commands.Cog, name='chess'):
	def __init__(self, bot):
		self.bot = bot
		self.games = {}
		self.elo = None
		self.hist = None
		self.load()
	
	def load(self):
		obj = load_json(settings.chess.path)
		self.elo = obj.get('elo')
		self.hist = obj.get('hist')
		return self 

	def save(self):
		save_json(settings.casino.path,
			{
				'elo': self.elo,
				'hist': self.hist
			}
		)
		return self 

	@commands.group()
	async def chess(self, ctx):
		if not ctx.invoked_subcommand:
			pass

	@chess.command()
	async def match(self, ctx, fen: Optional[str]=None):
		if fen is None:
			fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
		_log.info("New chess game FEN=%s", fen)
		th = await ctx.channel.create_thread(name=fen, invitable=True)
		view = discord.ui.View()	
		select = _userselect(ctx.author)
		view.add_item(select)
		await th.add_user(ctx.author)
		await th.send(f"**Select your opponent**", view=view)
		opponent, response = await select.future
		await th.add_user(opponent)
#		await th.purge()

		game = Engine.fromfen(fen)
		buff = BytesIO()
		img = generate_img(images=Images, game=game, fp=buff)
		buff.seek(0)
		await response.send_message(file=discord.File(buff, filename='board.png', description=game.tofen()))

		self.games[th.id] = {'th': th, 'game': game}
	
	@commands.Cog.listener()
	async def on_message(self, message):
		obj = self.games.get(message.channel.id, None)
		if not obj: return
		game = obj.get('game')

		if len(message.content) == 2:
			try:
				src = game.fromalpha(message.content) 
			except AssertionError as e:
				return
			try:
				moves = game.moves(src)
			except AssertionError as e:
				return await message.reply(e)
			if not moves:
				return await message.reply(f"No possible moves for {message.content}")
			buff = BytesIO()
			img = generate_img(images=Images, game=game, fp=buff, selected=message.content)
			buff.seek(0)
			await message.delete()
			return await obj['th'].send(file=discord.File(buff, filename='board.png', description=game.tofen()))
		if len(message.content) == 4:
			try:
				src, dst = message.content[:2], message.content[2:]
				if isinstance(src, str): src = Engine.fromalpha(src) 
				if isinstance(dst, str): dst = Engine.fromalpha(dst) 
			except AssertionError as e:
				return
			piece = None
			if game.board[src[::-1]] % (1 << 3) == Pieces.PAWN.value:
				if (game.turn == 'w' and dst[1] == 7) or (game.turn == 'b' and dst[1] == 0):
					view = discord.ui.View()
					select = _promoteselect(message.author)
					view.add_item(select)
					await obj['th'].send('**Select what to promote to**', view=view)
					piece, response = await select.future
					piece = Pieces(int(piece))
					await response.send_message(f"OK", ephemeral=True)
			try:
				game = game.move(src, dst, promote=piece)
			except AssertionError as e:
				return await message.reply(e)
			obj['game'] = game

			buff = BytesIO()
			img = generate_img(images=Images, game=game, fp=buff, lastmove=message.content)
			buff.seek(0)
			return await obj['th'].send(file=discord.File(buff, filename='board.png', description=game.tofen()))
		if message.content == "FEN":
			return await message.reply(game.tofen())
		await obj['th'].edit(name=game.tofen())



async def setup(bot):
	await bot.add_cog(Chess(bot))
