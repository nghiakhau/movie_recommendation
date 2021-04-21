from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as f:
    readme = f.read()

setup(
    name='movie_recommendation',
    version='0.1.0',
    description='Build a movie recommendation system on MovieLens dataset',
    packages=find_packages()
)