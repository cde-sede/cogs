from discord.ext import commands
import discord
import requests
import re
import settings

import logging
import json
import itertools

_log = logging.getLogger(__name__)

urlcheck = re.compile(r'(?<=\/)(?:[\w\d]+(?!\/)(?=.py$))')

def save_json(path, obj):
	with open(path, 'w', encoding='utf8') as f:
		json.dump(obj, f, indent=4)

def load_json(path):
	with open(path, 'r', encoding='utf8') as f:
		data = json.load(f)
	return data

def cog_basepath(cog):
	return cog.split('.')[-1]

class Loader(commands.Cog, name='loader'):
	class LoaderException(Exception): pass
	class _cog:
		LOCAL = 0b0
		REMOTE = 0b1

		@classmethod
		def getlocal(cls, param):
			try:
				assert (settings.ext_folder / f'{param}.py').exists()
				return cls(name=param, src=cls.LOCAL)
			except AssertionError:
				return None
		@classmethod
		def getremote(cls, param):
			try:
				return cls(url=param, src=cls.REMOTE)
			except AssertionError:
				return None

		def _localstr(self):
			return '.'.join([*settings.ext_folder.parts, self.name])

		def __init__(self, *, name=None, src=LOCAL, url=None):
			self.src = src
			self.url = None
			self.name = None
			if src & self.REMOTE:
				assert bool(url), 'invalid parameters' 
				self.url = url
				self.name = urlcheck.search(self.url) 
				assert bool(self.name)
				self.name = self.name.group()
				self.used = False
			else:
				assert bool(name), 'invalid parameters' 
				assert (settings.ext_folder / f'{name}.py').exists()
				self.name = name
			self.realpath = settings.ext_folder / f'{self.name}.py'
			self.path = self._localstr()
		
		def __repr__(self):
			return f"{self.name}:{self.path}"

		def get(self):
			if self.src & self.REMOTE:
				self.fetch_cog()
			else:
				pass
			return self.path	

		def fetch_cog(self):
			_log.info("Fetching cog %s (url %s)", self.name, self.url)
			with open(self.realpath, 'wb') as f: 
				r = requests.get(self.url, headers={'Cache-Control': 'no-cache'})
				if r.status_code != 200:
					_log.error("Error while fetching %s", self.name)
					raise Loader.LoaderException()
				f.write(r.content)

	class ExtConverter(commands.Converter):
		async def convert(self, ctx: commands.Context, argument: str):
			if (c := Loader._cog.getlocal(argument)):
				return c
			if (c := Loader._cog.getremote(argument)):
				return c
			raise commands.BadArgument(f'No cog {argument}')

	def __init__(self, bot):
		self.bot = bot

	def load_util(self, cog: _cog):
		_log.info("Loading %s", cog)
		cog.get()
		if cog.src & self._cog.REMOTE:
			cog.used = True
		self.bot.extcogs[cog.name] = cog
		self.bot.load_extension(cog.path)

	@commands.group()
	@commands.has_permissions(administrator=True)
	async def ext(self, ctx):
		if ctx.invoked_subcommand:
			return
		else:
			pass # TODO send help
		
	@ext.command()
	async def load(self, ctx, *, cog: ExtConverter):
		if cog.src & self._cog.REMOTE:
			await ctx.send('Please fetch before loading')
		else:
			path = cog.get()
			self.bot.extcogs[cog.name] = cog
			cog.used = True
			self.bot.load_extension(path)
			await ctx.send(f"Loaded {cog}!")

	@ext.command()
	async def unload(self, ctx, *, cog: ExtConverter):
		path = cog.get()
		self.bot.unload_extension(path)
		if seelf.bot.extcogs.get(cog.name).src == self._cog.REMOTE:
			cog.realpath.unlink(missing_ok=True)
		self.bot.extcogs.pop(cog.name)
		await ctx.send(f"Unloaded {cog}")

	@ext.command()
	async def reload(self, ctx, *, cog: ExtConverter):
		path = cog.get()
		self.bot.reload_extension(path)
		await ctx.send(f"Reloaded {cog}")

	@ext.command()
	async def fetch(self, ctx, *, cog: ExtConverter):
		if cog.src & self._cog.REMOTE:
			_log.info(f"Fetching {cog}")
			path = cog.get()
			self.bot.extcogs[cog.name] = cog
			await ctx.send(f"Fetched {cog}!")

	@ext.command()
	async def list(self, ctx):
		await ctx.send(f'{self.bot.extcogs}')

	@commands.group()
	@commands.has_permissions(administrator=True)
	async def autoload(self, ctx):
		if ctx.invoked_subcommand:
			return
		else:
			pass # TODO send help

	@autoload.command()
	async def add(self, ctx, cog: str):
		print(self.bot.extcogs)
		if self.bot.extcogs.get(cog, None) is None:
			return await ctx.send(f"`{cog}` is currently not a cog")
		data = load_json(settings.loader.path)
		obj = self.bot.extcogs.get(cog, None) 

		data.append(
			{
				"name": obj.name,
				"src": '1' if obj.src == self._cog.REMOTE else '0',
				"url": obj.url,
			})

		save_json(settings.loader.path, data)
		await ctx.send(f"Added `{cog}`")

	@autoload.command()
	async def edit(self, ctx, cog: str, key: str, value: str):
		if self.bot.extcogs.get(cog, None) is None:
			return await ctx.send(f"`{cog}` is currently not a cog")
		if key != 'src' and key != 'url':
			return await ctx.send(f"`{key}` is not a valid key. Use `src` or `url`")

		data = load_json(settings.loader.path)
		*cog_ ,= filter(lambda x: cog == x['name'], data)
		if len(cog_) == 0:
			return await ctx.send(f"`{cog}` is not in autoload")
		if len(cog_) > 1:
			return await ctx.send(f"`{cog}` has duplicates in autoload")
			
		if key == 'src':
			value = int(value)
		cog_[0][key] = value
		save_json(settings.loader.path, data)
		await ctx.send(f"Edited `{cog}`")

	@autoload.command()
	async def delete(self, ctx, name: str, force: bool=False):
		data = load_json(settings.loader.path)
		*matches ,= filter(lambda x: cog == x['name'], data)
		if len(matches) == 0:
			return await ctx.send(f"`{cog}` is not in autoload")


		if force == False:
			if len(matches) != 1:
				return await ctx.send(f"`{cog}` has duplicates in autoload, use `force=True` to delete them all")


		*remainder ,= filter(lambda x: cog != x['name'], data)
		save_json(settings.loader.path, remainder)
		await ctx.send(f"Deleted {'' if len(matches) == 1 else 'all '}`{cog}`")

	def _autoload(self):
		# for backwards compatibility's sake, second step shouldn't be used anymore
		# but data syntax for how modules should be loaded needs to stay consistent 
		# across legacy and current code
		data = load_json(settings.loader.path)
		for ext in itertools.chain(settings.second_step_ext, data):
			try:
				cog = self._cog(**ext)
				self.load_util(cog)
			except AssertionError as e:
				_log.error("AssertionError Error while loading `%s`, halt", ext['name'])
				raise # TODO proper error handling
			except Loader.LoaderException as e:
				_log.error("LoaderException Error while loading `%s`, skipped", ext['name'])
				pass


def setup(bot):
	bot.add_cog(Loader(bot))
