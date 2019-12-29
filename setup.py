from setuptools import find_packages, setup

setup(name='mocap_processing',
      version='0.1',
      description='',
      url='https://github.com/fairinternal/mocap_processing',
      author='FAIR Pittsburgh',
      author_email='dgopinath@fb.com',
      install_requires = [
      'basecode @ git+https://github.com/Jungdam/basecode#egg=basecode',
      'jupyter',
      'numpy',
      'pillow',
      'scipy',
      'PyOpenGL',
      ],
      packages=find_packages(),
      zip_safe=False
)
