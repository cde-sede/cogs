from discord.ext import commands 

class Print(commands.Cog, name='print'):
	def __init__(self, bot):
		self.bot = bot

	@commands.command()
	async def print(self, ctx, arg: str):
		await ctx.send(arg)

async def setup(bot):
	await bot.add_cog(Print(bot))
	
async def teardown(bot):
	pass
