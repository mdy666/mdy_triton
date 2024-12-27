from setuptools import setup, find_packages

setup(
    name="mdy_triton",
    version="0.1.0",  # 版本号
    author="mdy",  # 作者名字
    author_email="1670016147@qq.com",
    description="A package for core and replace_kernel functionality",  # 包描述
    long_description=open("README.md").read(),  # 长描述（通常从 README.md 读取）
    long_description_content_type="text/markdown",  # 长描述格式
    packages=find_packages(),  # 自动查找包
    install_requires=[  # 依赖项
        "transformers",
        "torch",
        "triton",
        # 添加其他依赖项
    ],
    classifiers=[  # 分类信息
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Python 版本要求
)