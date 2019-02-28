Conformal mappings and hyperbolic tessalations with Python
==========================================================

.. image:: https://badge.fury.io/py/confmap.svg
    :target: PyPI_
    :alt: ConfMap page on the Python Package Index

Python classes for transformations of pictures and videos (with moviepy_) using conformal mappings of the complex plane and hyperbolic tessalations of Poincaré's disk.

Author : François Coulombeau

Example
-------

In this example, we open an image and build a $+\infty;6;4;6$ hyperbolic tessalation :

.. code:: python
	import confmap as cm
	import numpy as np
	
	im = cm.HyperbolicTiling('./Sources/Reflets.jpg',0,'./Exports/',600,600)
	
	im.transform(sommets=(np.inf,6,4,6),nbit=20,backcolor=[255,255,255])

which leeds to this image :

.. image:: https://github.com/FCoulombeau/confmap/blob/master/examples/Reflets-0.jpg
    :align: center
    :alt: [tessalation]

Installation
------------

ConfMap depends on the Python modules Numpy_, `Matplotlib`_ and moviepy_ which will be automatically installed during ConfMap's installation. The software FFMPEG should be automatically downloaded/installed (by imageio) during your first use of MoviePy (installation will take a few seconds). If you want to use a specific version of FFMPEG, see Moviepy's documentation.

**Installation by hand:** download the sources, either from PyPI_ or, if you want the development version, from GitHub_, unzip everything into one folder, open a terminal and type:

.. code:: bash

    $ (sudo) python setup.py install

**Installation with pip:** if you have ``pip`` installed, just type this in a terminal:

.. code:: bash

    $ (sudo) pip install moviepy

Maintainers
-----------

- Oioi_ (owner)


.. ConfMap links
.. _documentation: http://zulko.github.io/moviepy/
.. _`download ConfMap`: https://github.com/FCoulombeau/confmap

.. Websites, Platforms
.. _PyPI: https://pypi.python.org/pypi/confmap
.. _GitHub: https://github.com/FCoulombeau/confmap

.. Software, Tools, Libraries
.. _Numpy: http://www.scipy.org/install.html
.. _`Matplotlib`: https://matplotlib.org/
.. _moviepy : https://github.com/Zulko/moviepy
.. _imageio: http://imageio.github.io/
.. _ffmpeg: http://www.ffmpeg.org/download.html
.. _ImageMagick: http://www.imagemagick.org/script/index.php
.. _`Sphinx`: http://www.sphinx-doc.org/en/master/setuptools.html

.. People
.. _Oioi: https://github.com/FCoulombeau