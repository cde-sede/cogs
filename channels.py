from discord.ext import commands 
import settings
import asyncio

class Channels(commands.Cog, name='channels'):
	def __init__(self, bot):
		self.bot = bot
		self.channels = {}
		self.category = None


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

		def check(member, before, after):
			return member == ctx.author

		try:
			await self.bot.wait_for('voice_state_update', check=check, timeout=settings.channels.timeout)
		except asyncio.TimeoutError:
			return await channel.delete()
		self.channels[ctx.author.id] = channel

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
				return member == m
			try:
				await self.bot.wait_for('voice_state_update', check=check, timeout=settings.channels.timeout)
			except asyncio.TimeoutError:
				self.channels.pop(k)
				return await before.channel.delete()



async def setup(bot):
	await bot.add_cog(Channels(bot))
	
