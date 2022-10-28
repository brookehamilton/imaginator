from setuptools import setup

setup(
    name='imaginator',
    version='0.1.0',
    description='Bringing kids\' artwork to life using Stable Diffusion',
    url='https://github.com/brookehamilton/imaginator',
    author='Brooke Hamilton',
    author_email='brookehamilton@gmail.com',
    license='BSD 2-clause',
    packages=['imaginator'],
    install_requires=[
                    'click',
                    'diffusers',
                    'huggingface-hub',
                    'numpy',
                    'Pillow',
                    'scipy',
                    'tokenizers',
                    'torch',
                    'transformers',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
