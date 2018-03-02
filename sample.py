from ILAMB.Variable import Variable

# pick LE and a model

obs = Variable(...)
mod = Variable(...)

def TimeSeries(obs,mod,site_id):
    pass
    

def Wavelet(obs,mod,site_id):
    pass

for site in sites:
    Timeseries(obs,mod,site)
    Wavelet   (obs,mod,site)
