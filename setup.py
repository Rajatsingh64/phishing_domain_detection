from setuptools import find_packages , setup

hyphen_e_dot="-e ."
requirements_files="requirements.txt"

def get_requirements():
    #getting all requirements as list 
    with open(requirements_files) as require_files:
        requirement_list=require_files.readlines()

    requirements_list=[]

    for require_name in requirement_list:
        striped_names=require_name.strip() #removing spaces 

        if striped_names!=hyphen_e_dot: 
            requirements_list.append(striped_names) #removing -e . from requirements while creating src package
    
    return requirements_list
       

setup(
       name="phishing" ,
       author="Rajat Singh",
       author_email="rajat.k.singh64@gmail.com" , 
       packages=find_packages() ,
       version="0.1.0" ,
       install_requires=get_requirements()
)