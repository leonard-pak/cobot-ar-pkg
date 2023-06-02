import os
from setuptools import setup
from glob import glob

package_name = 'cobot_ar_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*config.yaml')),
        (os.path.join('share', package_name, 'config'),
         glob('config/hand_landmarker.task')),
        (os.path.join('share', package_name, 'config'),
         glob('config/calibration_data_static.json')),
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
            'frame_processing = cobot_ar_pkg.frame_processing_node:main',
            'camera = cobot_ar_pkg.camera_node:main',
            'calibration = cobot_ar_pkg.camera_calibration:main'
        ],
    },
)
