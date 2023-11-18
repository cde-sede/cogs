from discord.ext import commands 
import settings
import asyncio
import logging
import json

_log = logging.getLogger(__name__)

def save_json(path, obj):
	with open(path, 'w', encoding='utf8') as f:
		json.dump(obj, f, indent=4)

def load_json(path):
	with open(path, 'r', encoding='utf8') as f:
		data = json.load(f)
	return data

class Channels(commands.Cog, name='channels'):
	def __init__(self, bot):
		self.bot = bot
		self.channels = {}
		self.category = None
	
	def load(self):
		self.channels = {}
		for author, channel in load_json(settings.channels.path).items():
			obj = self.bot.get_channel(channel)
			if obj:
				self.channels[author] = obj
	
	def save(self):
		save_json(settings.channels.path,
				{k:v.id for k,v in self.channels.items()})
	
	@commands.group()
	async def channel(self, ctx):
		pass

	@channel.command()
	async def new(self, ctx, name: str):			# TODO log inside reason
		"""Creates a new temporary channel.

		The channel gets deleted if it is inactive for 30secs or more.
		"""
		if self.category is None:
			for c in ctx.guild.categories:
				if c.name == 'temp':
					self.category = c
					break
		if self.channels.get(ctx.author.id, None) is not None:
			return await ctx.send("Each user can only create one channel at a time.")
		channel = await ctx.guild.create_voice_channel(name, category=self.category, position=0)
		await ctx.send(f'Channel was created. Join within {settings.channels.timeout} secs or the channel will be deleted due to inactivity')

		self.channels[ctx.author.id] = channel
		self.save()

		def check(member, before, after):
			return member == ctx.author and after.channel == channel

		try:
			await self.bot.wait_for('voice_state_update', check=check, timeout=settings.channels.timeout)
		except asyncio.TimeoutError:
			await channel.delete()
			self.channels.pop(ctx.author.id)
			self.save()
	
	@channel.command()
	async def rename(self, ctx, name: str):
		if self.channels.get(ctx.author.id, None) is None:
			return await ctx.send("You do not have a channel associated with you")
		await self.channels.get(ctx.author.id).edit(name=name)
		
	@commands.Cog.listener()
	async def on_voice_state_update(self, member, before, after):
		if before.channel == None:
			return
		for k,v in self.channels.items():
			if v == before.channel:
				break
		else:
			return
		if len(before.channel.members) == 0:
			def check(m, before, after):
				return v == after.channel
			try:
				await self.bot.wait_for('voice_state_update', check=check, timeout=settings.channels.timeout)
			except asyncio.TimeoutError:
				self.channels.pop(k)
				self.save()
				return await before.channel.delete()

	async def ready(self):
		self.load()

		async def _run(k, channel):
			if len(channel.members) != 0:
				return True
			def check(member, before, after):
				return after.channel == channel
			try:
				await self.bot.wait_for('voice_state_update', check=check, timeout=settings.channels.timeout)
			except asyncio.TimeoutError:
				self.channels.pop(k)
				self.save()
				return await channel.delete()

		if self.channels:
			_log.info("Resuming channel monitoring for %d channels", len(self.channels))
			await asyncio.gather(*[
					_run(k, c) for k,c in self.channels.items()
				])	
	
	async def cog_after_invoke(self, ctx):
		self.save()

def setup(bot):
	bot.add_cog(Channels(bot))
