from setuptools import find_packages, setup

setup(name='mocap_processing',
      version='0.1',
      description='',
      url='https://github.com/fairinternal/mocap_processing',
      author='FAIR Pittsburgh',
      author_email='dgopinath@fb.com',
      install_requires = [
      'basecode @ git+https://github.com/Jungdam/basecode#egg=basecode',
      'amass @ git+https://github.com/nghorbani/amass#egg=amass',
      'jupyter',
      'matplotlib',
      'numpy',
      'pillow',
      'pyglet @ git+https://github.com/mmatl/pyglet#egg=pyglet',
      'scipy',
      'torch',
      'tqdm',
      'PyOpenGL==3.1.0',
      ],
      packages=find_packages(),
      zip_safe=False
)
