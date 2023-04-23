from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)-> List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name='Machine Learning Project',
    version ='0.0.1',
    author ='Mohsin',
    author_email ='mohsinparmar98@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt')


)