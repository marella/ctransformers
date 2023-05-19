from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

name = 'ctransformers'

setup(
    name=name,
    version='0.1.2',
    description=
    'Python bindings for the Transformer models implemented in C/C++ using GGML library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ravindra Marella',
    author_email='mv.ravindra007@gmail.com',
    url='https://github.com/marella/{}'.format(name),
    license='MIT',
    packages=[name],
    package_data={name: ['lib/*/*.so', 'lib/*/*.dll', 'lib/*/*.dylib']},
    install_requires=[
        'huggingface-hub',
    ],
    extras_require={
        'tests': [
            'pytest',
        ],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='{} transformers ai llm'.format(name),
)
