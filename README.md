# HyperMapper

[![Build Status](https://travis-ci.com/luinardi/hypermapper.svg?branch=master)](https://travis-ci.com/luinardi/hypermapper)

## Software
See the wiki for the [quick start guide](https://github.com/luinardi/hypermapper/wiki).

## Abstract
HyperMapper is a multi-objective black-box optimization tool based on Bayesian Optimization. 

HyperMapper was succesfully applied to real-world problems involving design search spaces with trillions of possible design choices. In particular it was applied to: 
1) computer vision and robotics,
2) database management systems (DBMS) parameters configuration, 
3) domain-specific language (DSL) compilers and hardware design. 

To learn about the core principles of HyperMapper refer to the papers section in the bottom. 

## Contact and Info
For any questions please contact [Luigi Nardi](lnardi@stanford.edu).

## License
HyperMapper is distributed under the MIT license. More information on the license can be found [here](https://github.com/luinardi/hypermapper/blob/master/LICENSE).

## People
### Main Contributors
    Luigi Nardi, Assistant Professor, Lund University, and Research Scientist, Stanford University
    Artur Souza, Ph.D. Student, Federal University of Minas Gerais
### Other Contributors    
    Bruno Bodin, Assistant Professor, National University of Singapore 

## Papers
If you use HyperMapper in scientific publications, we would appreciate citations to the following paper: 

Practical design space exploration (MASCOTS 2019) - introducing HyperMapper principles and application to hardware design space exploration: 
```latex
@inproceedings{nardi2019practical,
  title={Practical design space exploration},
  author={Nardi, Luigi and Koeplinger, David and Olukotun, Kunle},
  booktitle={2019 IEEE 27th International Symposium on Modeling, Analysis, and Simulation of Computer and Telecommunication Systems (MASCOTS)},
  pages={347--358},
  year={2019},
  organization={IEEE}
}
```

### Other papers about HyperMapper and its applications 

Spatial: A Language and Compiler for Application Accelerators (PLDI 2018) - conference paper on the application of HyperMapper to the Spatial programming language:
```latex
    @inproceedings{koeplinger2018spatial,
    title={Spatial: a language and compiler for application accelerators},
    author={Koeplinger, David and Feldman, Matthew and Prabhakar, Raghu and Zhang, Yaqi and Hadjis, Stefan and Fiszel, Ruben and Zhao, Tian and Nardi, Luigi and Pedram, Ardavan and Kozyrakis, Christos and others},
    booktitle={Proceedings of the 39th ACM SIGPLAN Conference on Programming Language Design and Implementation},
    pages={296--311},
    year={2018},
    organization={ACM}
    }
```

Algorithmic Performance-Accuracy Trade-off in 3D Vision Applications Using HyperMapper (iWAPT 2017) - workshop on the application of HyperMapper to ElasticFusion and KinectFusion computer vision applications:
```latex
    @inproceedings{nardi2017algorithmic,
    title={Algorithmic performance-accuracy trade-off in 3D vision applications using hypermapper},
    author={Nardi, Luigi and Bodin, Bruno and Saeedi, Sajad and Vespa, Emanuele and Davison, Andrew J and Kelly, Paul HJ},
    booktitle={Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2017 IEEE International},
    pages={1434--1443},
    year={2017},
    organization={IEEE}
    }
```

Integrating algorithmic parameters into benchmarking and design space exploration in 3D scene understanding (PACT 2016) -  conference paper applying an early version of HyperMapper to 3D computer vision:
```latex
    @inproceedings{bodin2016integrating,
    title={Integrating algorithmic parameters into benchmarking and design space exploration in 3D scene understanding},
    author={Bodin, Bruno and Nardi, Luigi and Zia, M Zeeshan and Wagstaff, Harry and Sreekar Shenoy, Govind and Emani, Murali and Mawer, John and Kotselidis, Christos and Nisbet, Andy and Lujan, Mikel and others},
    booktitle={Proceedings of the 2016 International Conference on Parallel Architectures and Compilation},
    pages={57--69},
    year={2016},
    organization={ACM}
    }
```

Application-oriented design space exploration for SLAM algorithms (ICRA 2017) - conference paper on the application of HyperMapper to robotics: 
```latex
    @inproceedings{saeedi2017application,
    title={Application-oriented design space exploration for SLAM algorithms},
    author={Saeedi, Sajad and Nardi, Luigi and Johns, Edward and Bodin, Bruno and Kelly, Paul HJ and Davison, Andrew J},
    booktitle={Robotics and Automation (ICRA), 2017 IEEE International Conference on},
    pages={5716--5723},
    year={2017},
    organization={IEEE}
    }
```
