from distutils.core import setup, Extension

module = Extension('_libiq',
                   sources=['src/libiq_wrap.cxx', 'src/converter.cpp', 'src/analyzer.cpp'],
                   include_dirs=[
                         '/usr/local/include',
                         '/usr/local/include/sigmf',
                         './libs/libsigmf/external/flatbuffers/include',
                         './libs/libsigmf/external/json/include',
                         '/usr/local/include/opencv4'
                         ],
                   libraries=[
                         'matio',
                         'fftw3',
                         'opencv_calib3d',
                         'opencv_core',
                         'opencv_dnn',
                         'opencv_features2d',
                         'opencv_flann',
                         'opencv_gapi',
                         'opencv_highgui',
                         'opencv_imgcodecs',
                         'opencv_imgproc',
                         'opencv_ml',
                         'opencv_objdetect',
                         'opencv_photo',
                         'opencv_stitching',
                         'opencv_video',
                         'opencv_videoio'
                         ],
                   library_dirs=['/usr/local/lib']
                   )

setup(name='libiq',
      version='0.1',
      author="TUO NOME",
      description="""Esempio semplice di swig dalla documentazione""",
      ext_modules=[module],
      py_modules=["libiq"],
      )