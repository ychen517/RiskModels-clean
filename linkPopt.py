
import platform
import os
import os.path

if __name__ == '__main__':
   poptversion = '2.25.SNAPSHOT'

   libcmap = {('Ubuntu', '12.04'): 'glibc2.15-gcc4.6', ('Ubuntu', '10.10'): 'glibc2.11-gcc4.4', ('Ubuntu', '10.04'): 'glibc2.11-gcc4.4',
              ('Ubuntu', '8.04'): 'glibc2.7-gcc4.2', }

   def readLSBRelease():
      if os.path.exists('/etc/lsb-release'):
         infile = open('/etc/lsb-release')
         return dict(line.strip().split('=') for line in infile)
      else:
         raise Exception('Unsupported OS, does not have /etc/lsb-release')

   hosttype = platform.system()

   if hosttype == 'Linux':
      lsbinfo = readLSBRelease()
      libcstr = libcmap.get((lsbinfo['DISTRIB_ID'],lsbinfo['DISTRIB_RELEASE']), None)
      if libcstr is None:
         raise Exception('Unknown OS version %s' % str(lsbinfo))
      arch = platform.architecture()
      if arch[0] == '32bit':
         arch = 'i386'
      elif arch[0] == '64bit':
         arch = 'amd64'
      else:
         raise Exception('Unknown architecture %s' % arch)

      arch = arch + '-' + libcstr

      path = os.path.join(os.sep, 'axioma', 'products', 'current', 'python-popt', poptversion, arch)
      if not os.path.exists(path):
         raise Exception('Cannot find path %s' % path)
      if os.path.exists('popt'):
         if os.path.islink('popt'):
            os.remove('popt')
         else:
            raise Exception('Directory/file named popt already exists')

      os.symlink(path, 'popt')
      import popt
   else:
      raise Exception('Unsupported system: %s' % hosttype)


