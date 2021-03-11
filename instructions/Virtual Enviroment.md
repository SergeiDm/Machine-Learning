Recommendation for a Python project workspace[^1]:

├── requirements.txt     ---> Project dependencies file

├── src                  ---> Source code goes here

├── tests                ---> Test code goes here

└── venv                 ---> Virtual environment files

## Steps for creating virtual environments: 
> 0. **Install jupyterlab**
> `pip install jupyterlab`
> 1.  **Go to the project directory**
> `cd C:\Users\user_name\my_project`
> 2. **Create virtual environment**
> `py -3.7 -m venv venv`
where 3.7 - specific version of python
> 3. **Activate virtual environment**
`C:\Users\user_name\my_project\venv\Scripts\activate`
> 4. **Install packages[^2]**
`pip install <package_name>`

    If there is error: pip install fails with “connection error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed”, 
    the following command should be used: `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package_name>`

> 5. **Install ipykernel into new environment** 
`pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ipykernel`

> 6. **Install new kernel**
`ipython kernel install --user --name=<Name>`

## Some operations:
> * **Listing kernels**
`jupyter kernelspec list`
> * **Removing kernel**
`jupyter kernelspec remove <kernel-name>`
> * **Changing kernel name**:  
    1. Open file 'kernel.json' in kernel location folder  
    2. Rename property 'display_name'  

## Install Packages (alternatives way)  
`git clone https://github.com/<name library>.git`  
`cd <name library>`  
`pip install .`  
In this example 'requirements.txt' can be changed that is suitable for the case when two libraries require each other.

**References**  
[^1]: [Python Setup: The Definitive Guide](https://www.datacamp.com/community/tutorials/python-developer-set-up)  
[^2]: [Installing Packages](https://packaging.python.org/tutorials/installing-packages/)  
