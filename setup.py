from setuptools import setup, find_packages

package_name = 'serafin'

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req:
    requirements = req.read().splitlines()

setup( name                          = package_name
     , version                       = '0.0.1'
     , author                        = 'Yannick Uhlmann'
     , author_email                  = 'augustunderground@pm.me'
     , description                   = 'Single-Ended Operational Amplifier Characterization'
     , long_description              = long_description
     , long_description_content_type = 'text/markdown'
     , url                           = 'https://github.com/augustunderground/serafin'
     , packages                      = find_packages()
     , classifiers                   = [ 'Development Status :: 2 :: Pre-Alpha'
                                       , 'Programming Language :: Python :: 3'
                                       , 'Operating System :: POSIX :: Linux' ]
     , python_requires               = '>=3.9'
     , install_requires              = requirements
     #, entry_points                  = { 'console_scripts': [ 'FIXME' ]}
     , package_data                  = { '': ['resource/testbench.scs']}
     , include_package_data          = True
     , )
