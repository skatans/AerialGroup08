from setuptools import setup

package_name = 'opencv_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sandra',
    maintainer_email='sandra@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'img_sub = opencv_package.image_subscriber:main',
            'img_det = opencv_package.object_detection:main',
            'gate_det = opencv_package.gate_detection:main',
        ],
    },
)
