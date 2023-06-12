import os
from setuptools import setup
from glob import glob

package_name = 'cobot_ar_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name+'.utils'],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*config.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.task')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonard Pak',
    maintainer_email='leopak2000@gmail.com',
    description='TODO: Package description',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frames_processing = cobot_ar_pkg.frames_processing_node:main',
            'camera = cobot_ar_pkg.camera_node:main',
            'camera_calibration = cobot_ar_pkg.camera_calibration_node:main',
            'point_processing = cobot_ar_pkg.point_processing_node:main'
        ],
    },
)
