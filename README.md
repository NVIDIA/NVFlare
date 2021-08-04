# NVFlare

This repo currently contains example codes only.  It intends for users to get a tast of NVFlare by running
three simple exercises, based on three different computation frameworks, namely PyTorch, Numpy and Tensorflow 2.

The document for this repo can be find in doc/source folder.  It is in Sphinx RST format and is not
intended to view directly.  To generate the HTML version of the document, which is more suitable to
read, please do the following (assume you are in the hello_nvflare directory):

  * cd doc
  * python3 -m pip install -r requirements.txt
  * make html

The requirements.txt inside doc folder is for dependencies required for document build process.  The
document itself describes how to install NVFlare and how to run those three examples.

The result html files are in doc/build/html.  You can open index.html in your own browser.

The project document page is available [here](https://nvidia.github.io/nvflare/).

NVFlare can be installed via Python pip install.  The PyPi page of NVFlare is available [here](https://pypi.org/project/nvflare/).