# Not a cog, don't try to load it
# it's basically a singleton that parses a settings file
# I'll add an example file later
# But it's basically python with some control over the
# structure of the final object
def _inc(lines):
	ext = {}
	for l in lines:
		if l[0] == '[' and l.strip()[-1] == ']':
			args = l.strip()[1:-1].split(':')
			lib = __import__(args[0])
			for i in args[1].split(','):
				if hasattr(lib, i):
					ext[i] = getattr(lib, i)
		else: break
	return l, ext

def _open():
	with open('settings.env', 'r', encoding='utf8') as f:
		file = f.readlines()
	lines = iter(file)
	l,ext = _inc(lines)
	r = [["settings", None, ext]]
	c = [l]
	for l in lines:
		if l[0] == '{' and l.strip()[-1] == '}':
			name = l.strip()[1:-1]
			r[-1][1] = ''.join(c)
			l,ext = _inc(lines)
			c = [l]
			r.append([name, None, ext])
		else:
			c.append(l)
	r[-1][1] = ''.join(c)
	return r

def _obj(n,c,l):
	globals()['l'] = l
	code = f"""
class {n}:
	locals().update(l)
	exec(compile('''{c}''', '{n}', 'exec'))
	for i in l: del locals()[i]
	"""
	return compile(code, '', 'exec')
	

if __name__ != '__main__':
	import sys
	class _settings:...
	if not any(isinstance(i, _settings) for i in sys.modules):
		f = _open()

		class settings(_settings):
			locals().update(f[0][2])
			exec(compile(f[0][1], 'settings', 'exec'))
			for n,c,l in f[1:]:
				exec(_obj(n,c,l))
			for i in f[0][2]:
				del locals()[i]

		sys.modules[__name__] = settings()
