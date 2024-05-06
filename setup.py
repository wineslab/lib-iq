from setuptools import setup, Extension

module = Extension('_libiq',
                   sources=['src/libiq_wrap.cxx', 'src/converter.cpp', 'src/analyzer.cpp'],
                   include_dirs=[
                         '/usr/local/include',
                         '/usr/local/include/sigmf',
                         './libs/libsigmf/external/flatbuffers/include',
                         './libs/libsigmf/external/json/include',
                         ],
                   libraries=[
                         'matio',
                         'fftw3'
                         ],
                   library_dirs=['/usr/local/lib']
                   )

setup(name='libiq',
      version='0.1',
      author="TUO NOME",
      description="""Esempio semplice di swig dalla documentazione""",
      ext_modules=[module],
      py_modules=["libiq", "src_python.spectrogram", "src_python.scatterplot"],
      )
