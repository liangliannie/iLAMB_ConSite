# PyE3SM

This Python Package, name PyE3SM, is established to comprehensively examine the land component of Energy Exascale Earth System Model (E3SM). The PyE3SM can be viewed as a visualization tool, which is able to provide integrated evaluation of E3SM performance for the individual variables and relationships among multiple variables. 


![Package Structure](./structure.png "Package Structure")


Mainly, PyE3SM is trying to evaluate the land models throught four aspects: Time Series, Cycle Means, Frequency, and Responses.
The package can be run directly under the environment of iLAMB, since it has been merged as a confrontation file of iLAMB.

```python 
python ConfSite2.py
python ConfSite3.py
```

Here 2, 3 defined different versions of the confrontation file, where ConfSite3.py include the usage of Pandas to further improve the accuacy and efficiency of the package.

Selective outputs of the package are listed below,

![Selective outputs](./site.png  "Selective outputse"=300x)
![Selective outputs](./time1.png "Selective outputs"=300x)
![Selective outputs](./cycle.png  "Selective outputs"=300x)
![Selective outputs](./taylor.png  "Selective outputs"=300x)
![Selective outputs](./wavelet.png "Selective outputs"=300x)

Overall, this can also be viewed as an extension for adding new metrics to iLAMB which works as the site-level component for land models' evaluations.

The example output of the package can be found below:
[Webpage CERES.html](http://volweb.utk.edu/~lli51/ol2/CERES.html)

Please cite our coming paper,

Liang Li, Jiafu Mao, Daniel M. Ricciuto, "Developing a site-level diagnostic package for the land component of E3SM"

