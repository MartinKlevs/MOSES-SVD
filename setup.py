from setuptools import setup

setup(
        name='MOSESSVD',
        version='1.0',
        maintainer="Martins Klevs",
        url="https://github.com/MartinKlevs/MOSES-SVD/",
        keywords=["svd", "streaming-svd", "online-svd"],
        packages=['MOSESSVD'],
        install_requires=["numpy", "scipy", "numba"],
        licence="Unlicence",
     )
