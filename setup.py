from setuptools import find_packages, setup
import os
import shutil
import sys
import torch
import warnings
from os import path as osp
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


version_file = 'mmdet3d/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    import sys
    if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        return locals()['short_version']
    else:
        return locals()['__version__']


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=None,
                  extra_include_path=[]):
    """Create a CUDA or C++ extension module with unified C++17 standard."""
    if extra_args is None:
        extra_args = []

    define_macros = []
    extra_compile_args = {'cxx': extra_args + ['-std=c++17', '-w']}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '--expt-relaxed-constexpr',
            '-std=c++17'
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse requirements file and ignore version pinning."""
    import re
    from os.path import exists

    def parse_line(line):
        if line.startswith('-r '):
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]
                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(fname):
            for info in parse_require_file(fname):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                platform_deps = info.get('platform_deps')
                if platform_deps:
                    parts.append(';' + platform_deps)
                yield ''.join(parts)

    return list(gen_packages_items())


def add_mim_extension():
    """Add symlink or copy for MIM files when installed in editable mode."""
    if 'develop' in sys.argv:
        mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        mode = 'copy'
    else:
        return

    filenames = ['tools', 'configs', 'model-index.yml']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'mmdet3d', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)
            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)
            if mode == 'symlink':
                os.symlink(osp.relpath(src_path, osp.dirname(tar_path)), tar_path)
            elif mode == 'copy':
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='mmdet3d',
        version=get_version(),
        description="OpenMMLab's next-generation 3D object detection platform",
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='MMDetection3D Contributors',
        author_email='zwwdev@gmail.com',
        keywords='computer vision, 3D object detection',
        url='https://github.com/open-mmlab/mmdetection3d',
        packages=find_packages(),
        include_package_data=True,
        package_data={'mmdet3d.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[
            make_cuda_ext(
                name='sparse_conv_ext',
                module='mmdet3d.ops.spconv',
                extra_include_path=[
                    os.path.abspath(
                        os.path.join(*'mmdet3d.ops.spconv'.split('.'), 'include/')
                    )
                ],
                sources=[
                    'src/all.cc', 'src/reordering.cc', 'src/reordering_cuda.cu',
                    'src/indice.cc', 'src/indice_cuda.cu',
                    'src/maxpool.cc', 'src/maxpool_cuda.cu'
                ]),
            make_cuda_ext(
                name='iou3d_cuda',
                module='mmdet3d.ops.iou3d',
                sources=['src/iou3d.cpp', 'src/iou3d_kernel.cu']),
            make_cuda_ext(
                name='voxel_layer',
                module='mmdet3d.ops.voxel',
                sources=[
                    'src/voxelization.cpp', 'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu', 'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu']),
            make_cuda_ext(
                name='roiaware_pool3d_ext',
                module='mmdet3d.ops.roiaware_pool3d',
                sources=['src/roiaware_pool3d.cpp', 'src/points_in_boxes_cpu.cpp'],
                sources_cuda=[
                    'src/roiaware_pool3d_kernel.cu',
                    'src/points_in_boxes_cuda.cu']),
            make_cuda_ext(
                name='ball_query_ext',
                module='mmdet3d.ops.ball_query',
                sources=['src/ball_query.cpp'],
                sources_cuda=['src/ball_query_cuda.cu']),
            make_cuda_ext(
                name='knn_ext',
                module='mmdet3d.ops.knn',
                sources=['src/knn.cpp'],
                sources_cuda=['src/knn_cuda.cu']),
            make_cuda_ext(
                name='assign_score_withk_ext',
                module='mmdet3d.ops.paconv',
                sources=['src/assign_score_withk.cpp'],
                sources_cuda=['src/assign_score_withk_cuda.cu']),
            make_cuda_ext(
                name='group_points_ext',
                module='mmdet3d.ops.group_points',
                sources=['src/group_points.cpp'],
                sources_cuda=['src/group_points_cuda.cu']),
            make_cuda_ext(
                name='interpolate_ext',
                module='mmdet3d.ops.interpolate',
                sources=['src/interpolate.cpp'],
                sources_cuda=['src/three_interpolate_cuda.cu',
                              'src/three_nn_cuda.cu']),
            make_cuda_ext(
                name='furthest_point_sample_ext',
                module='mmdet3d.ops.furthest_point_sample',
                sources=['src/furthest_point_sample.cpp'],
                sources_cuda=['src/furthest_point_sample_cuda.cu']),
            make_cuda_ext(
                name='gather_points_ext',
                module='mmdet3d.ops.gather_points',
                sources=['src/gather_points.cpp'],
                sources_cuda=['src/gather_points_cuda.cu'])
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
