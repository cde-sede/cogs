from discord.ext import commands
import discord
import settings
import asyncio

from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty

from yt_dlp import YoutubeDL

import logging
_log = logging.getLogger(__name__) 


class _md:
	_q = Queue()
	downloading = False
	def __new__(self, *args):
		n = object.__new__(self)
		n.future = asyncio.Future()
		n.future.add_done_callback(lambda _f: _md._next())
		_md._q.put(n)
		if not _md.downloading:
			_md._next()
		return n

	@classmethod
	def _next(cls):
		if cls._q.empty():
			return
		cls.downloading = True 
		n = cls._q.get()
		n()
		n.future.add_done_callback(lambda _f: setattr(cls, 'downloading', False))

class _download(_md):
	def __init__(self, song: 'Song'):
		self.song = song

	def __call__(self):
		asyncio.create_task(self.download())

	def done(self, v):
		self.song.done.set_result(v)
		self.future.set_result(v)
		_log.info('DONE %s', self.song.url)
		return v

	async def download(self):
		_log.info('Downloading %s', self.song.url)
		if (self.song.path.exists()):
			return self.done(True)
		with YoutubeDL(settings.music.YDL_OPT) as ydl:
			try:
				return self.done(ydl.download(self.song.url))
			except Exception:
				return self.done(False)
		return self.done(False)

@dataclass
class Song:
	query: str
	url: str=''
	title: str=''
	path: Path=None
	duration: int=None
	channel: discord.VoiceChannel=None
	looped: bool=False

	done: asyncio.Future=field(default_factory=asyncio.Future)

	def download(self):
		if self.done.done():
			return True
		self._d = _download(self)


class Music(commands.Cog, name='music'):
	def __init__(self, bot):
		self.bot: commands.Bot = bot
		self.queue: Queue = Queue()
		self.voice = None
		self.playing = False

	async def search(self, item: Song):
		# TODO handle other sources, probably defer the searching to a
		#	commandless cog
		_log.info('Searching %s', item.query)
		with YoutubeDL(settings.music.YDL_OPT) as ydl:
			info = ydl.extract_info(item.query, download=False)

		item.url		= info['original_url']
		item.title		= info['title']
		item.path		= settings.music.path / (settings.music.template % info)
		item.duration	= info['duration']
		return item

	async def stream(self):
		loop = asyncio.get_event_loop()

		try:
			if self.voice is not None and self.voice.is_playing():
				return
			_log.info("%s", str(self.queue.queue))
			song = self.queue.get(timeout=settings.music.disconnect_time)

			def _recur(error):
				# Called after the music played 
				if error:
					_log.warning("Error while stream recursion. Should never happen")
					raise error
				if song.looped:
					self.queue.put(song)
				loop.create_task(self.stream())

			song.download()
			await song.done

			if self.voice is None:
				self.voice = await song.channel.connect() 
			await self.voice.move_to(song.channel)	

			self.voice.play(
					discord.FFmpegPCMAudio(
							source=song.path,
							**settings.music.FFMPEG_OPT
						),
					after=_recur
				)
		except Empty:
			await self.disconnect()

	async def disconnect(self):
		if self.voice:
			if self.voice.is_playing():
				self.voice.stop()
			self.voice = await self.voice.disconnect()
		self.playing = False

	@commands.group(aliases=['m'])
	async def music(self, ctx):
		if not ctx.invoked_subcommand:
			pass

	@music.command()
	async def play(self, ctx, query: str):
		try:
			song = await self.search(Song(query))
		except Exception as e:
			raise
		song.channel = ctx.author.voice.channel
		self.queue.put(song)
		if not self.playing:
			self.playing = True
			asyncio.create_task(self.stream())

	@music.command()
	async def loop(self, ctx, query: str):
		try:
			song = await self.search(Song(query))
		except Exception as e:
			raise
		song.channel = ctx.author.voice.channel
		song.looped = True
		self.queue.put(song)
		if not self.playing:
			self.playing = True
			asyncio.create_task(self.stream())

	@music.command()
	async def skip(self, ctx):
		if self.voice:
			self.voice.stop()

	@music.command()
	async def leave(self, ctx):
		await self.disconnect()
		self.queue.queue.clear()

	@music.command()
	async def queue(self, ctx):
		...
		# TODO prettyprint queue
		
def setup(bot):
	bot.add_cog(Music(bot))
