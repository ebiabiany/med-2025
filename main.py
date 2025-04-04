import os
import sys
import time
import datetime

from tools import *
from PIL import Image
from scipy.spatial import distance as dist

from matplotlib.colors import ListedColormap
if os.name=='nt' and ('win' in sys.platform):
    import h5py as h5
    import netCDF4 as nc
    # import requests as req
    import kmedoids

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from sklearn.decomposition import PCA
#import requests as req
#from sklearn.neighbors import NearestNeighbors


# Loads data from a NetCDF or HDF5 file and returns it as a NumPy array.
def loadData(fname, vname):
    try:
        ds = nc.Dataset(fname)  # type: ignore
    except:
        ds = h5.File(fname, 'r')
    return np.asarray(ds[vname])

# Finds the indices and cropped vector for a 1D range based on start and end values.
def crop1d(dep, fin, vec):
    min1 = min(vec, key=lambda val: abs(val-dep))
    min2 = min(vec, key=lambda val: abs(val-fin))
    xdep, xfin = np.where(vec == min1), np.where(vec == min2)
    if xfin[0][0] < xdep[0][0]:
        tmp = xdep[0][0]
        xdep[0][0] = xfin[0][0]
        xfin[0][0] = tmp
    return xdep[0][0], xfin[0][0], vec[xdep[0][0]:xfin[0][0]+1]

# Crops a 2D array based on specified start and end indices for both dimensions.
def crop2d(xdep, xfin, ydep, yfin, vec2d):
    return np.asarray(vec2d[xdep:xfin+1, ydep:yfin+1])

# Loads ERA5 data for a specific day and variable, handling time slicing.
def loadEra5(fname, varname, numJour, nbj=0):
    if nbj == 0:
        nbj = len(loadData(fname, "time"))//4
    if numJour > nbj:
        print("il n'ya pas autant de jour dans ce mois max : ", nbj)
        quit()
    else:
        variable = loadData(fname, varname)
        return np.asarray(variable[4*numJour-2, :, :])

# Converts ERA5 data into an analysis matrix based on a specified box and time step.
def era2analyse(fname, vname, box, h, dt, nanValue=None):
    xdep, xfin, ydep, yfin = box
    ds = nc.Dataset(fname)  # type: ignore
    time = np.asarray(ds['time'])
    data = np.asarray(ds[vname])
    if nanValue != None:
        data[data == nanValue] = np.float64('nan')
    analyse = []
    for j in range(h-1, len(time), dt):
        analyse.append(data[j, xdep:xfin+1, ydep:yfin+1].flatten())
    return analyse

# Converts NOAA data into an analysis matrix based on a specified box.
def noaa2analyse(fname, vname, box):
    xdep, xfin, ydep, yfin = box
    analyse = []
    ds = nc.Dataset(fname)  # type: ignore
    time = np.asarray(ds['time'])
    data = np.asarray(ds[vname])
    for j in range(len(time)):
        analyse.append(data[j, xdep:xfin+1, ydep:yfin+1].flatten())
    return analyse

# Generates a daily analysis matrix for all days in a directory, based on the data type.
def Alldays_matrix(rep, vname, DataType, coo=[-66.25, 30., -20.25, 5.], save=False, nanValue=None, savepath="./"):
    matrix = []
    files = sorted(os.listdir(rep))
    lon = lat = xdep = xfin = ydep = yfin = lat_crop = lon_crop = 0
    if DataType == "TRMM":
        lon = loadData(rep+files[0], "lon")
        lat = loadData(rep+files[0], "lat")
        xdep, xfin, lon_crop = crop1d(coo[0], coo[2], lon)
        ydep, yfin, lat_crop = crop1d(coo[3], coo[1], lat)
    elif DataType == "ERA5":
        lon = loadData(rep+files[0], "longitude")
        lat = loadData(rep+files[0], "latitude")
        xdep, xfin, lat_crop = crop1d(coo[3], coo[1], lat)
        ydep, yfin, lon_crop = crop1d(
            (coo[0] < 0)*360+coo[0], (coo[2] < 0)*360+coo[2], lon)
    elif DataType == "NOAA":
        lon = loadData(rep+files[0], "lon")
        lat = loadData(rep+files[0], "lat")
        xdep, xfin, lat_crop = crop1d(coo[3], coo[1], lat)
        ydep, yfin, lon_crop = crop1d(
            (coo[0] < 0)*360+coo[0], (coo[2] < 0)*360+coo[2], lon)

    np.save(f"./MatricesAnalyse/coo-{vname}", {"lat": lat_crop, "lon": lon_crop})  # type: ignore
    for f in files:
        print(">>>", f)
        if DataType == "TRMM":
            if f.endswith('.nc4'):
                variable = loadData(rep+f, vname)
                if nanValue != None:
                    variable[variable == nanValue] = np.float64("nan")
                matrix.append(
                    crop2d(xdep, xfin, ydep, yfin, variable).flatten())
        elif DataType == "ERA5":
            if f.endswith('.nc'):
                matrix = matrix + \
                    era2analyse(rep+f, vname, (xdep, xfin,
                                ydep, yfin), 3, 4, nanValue)
        elif DataType == "NOAA":
            if f.endswith('.nc'):
                matrix = matrix + \
                    noaa2analyse(rep+f, vname, (xdep, xfin, ydep, yfin))
    if save:
        np.save(f"{savepath}Analyse{vname.upper()}", np.asarray(matrix))
    return np.asarray(matrix)

# Plots data on a map with optional cropping and saving functionality.
def plotData2(data, lon, lat, DataType, titre, coo=[-66.25, 30, -20.25, 5], nanValue=None, savename="exemple", savepath="./", estCrop=False):
    if DataType == "TRMM":
        xdep, xfin, lon_crop = crop1d(coo[0], coo[2], lon)
        ydep, yfin, lat_crop = crop1d(coo[3], coo[1], lat)
        if estCrop:
            data_crop = data
        else:
            data_crop = crop2d(xdep, xfin, ydep, yfin, data)

        if nanValue != None:
            data_crop[data_crop == nanValue] = np.float64("nan")

        x, y = np.meshgrid(lat_crop, lon_crop)
        plotData(data_crop, y, x, box=coo, title=titre,fname=savename, path=savepath)

    elif DataType == "ERA5":
        xdep, xfin, lat_crop = crop1d(coo[3], coo[1], lat)
        ydep, yfin, lon_crop = crop1d(
            (coo[0] < 0)*360+(coo[0]), (coo[2] < 0)*360+(coo[2]), lon)
        if estCrop:
            data_crop = data
        else:
            data_crop = crop2d(xdep, xfin, ydep, yfin, data)
        if nanValue != None:
            data_crop[data_crop == nanValue] = np.float64("nan")
        x, y = np.meshgrid(lon_crop-360, lat_crop)
        plotData(data_crop, x, y, box=coo, title=titre,fname=savename, path=savepath)
    elif DataType == "NOAA":
        xdep, xfin, lat_crop = crop1d(coo[3], coo[1], lat)
        ydep, yfin, lon_crop = crop1d((coo[0] < 0)*360+coo[0], (coo[2] < 0)*360+coo[2], lon)
        if estCrop:
            data_crop = data
        else:
            data_crop = crop2d(xdep, xfin, ydep, yfin, data)
        x, y = np.meshgrid(lon_crop-360, lat_crop)
        plotData(data_crop, x, y, box=coo, title=titre,fname=savename, path=savepath)

# Returns the index in the analysis matrix corresponding to a specific date.
def getIdByDate(date, anneeDep, anneeFin):
    d1 = datetime.date(anneeDep, 1, 1)
    split_date = date.split("-")
    if int(split_date[2]) > anneeFin:
        print("Date incorrecte.")
        quit()
    else:
        d2 = datetime.date(int(split_date[2]), int(
            split_date[1]), int(split_date[0]))

    return (d2-d1).days

# Loads a specific day from an analysis matrix based on a date or index.
def loadAnalyse(fname, anneeDep, anneeFin, DataType, nblat, nblon, date=None, id=0):
    if date != None:
        id = getIdByDate(date, anneeDep, anneeFin)
    data = np.load(fname)[id, :]
    data_reshape = data
    if DataType == "TRMM":
        data_reshape = np.reshape(data, (nblon, nblat))
    elif DataType == "ERA5":
        data_reshape = np.reshape(data, (nblat, nblon))
    elif DataType == "NOAA":
        data_reshape = np.reshape(data, (nblat, nblon))

    return data_reshape

# Creates binary masks from image files for a specified number of zones.
def CreateMasks(nb_masks, path="./", savepath="./", deg1=False):
    masks = []
    if deg1:
        path += "deg1/"
    for m in range(1, nb_masks+1):

        print(path)
        mask = np.asarray(Image.open(f'{path}zone'+str(m)+'.bmp'), dtype=float)
        mask[mask > 0] = 1
        mask[mask == 0] = np.float64('nan')
        masks.append(mask.flatten())
    masks = np.asarray(masks)
    np.save(f"{savepath}masks_etude", masks)

# Computes histogram bin edges for data, with optional precipitation-specific bins.
def getEdges(data, nbins=10, eps=1e-7, addZero=True, precip=False):
    data = np.delete(data, np.where(np.isnan(data)))
    flat_data = data.flatten()
    flat_data = np.delete(flat_data, np.where(np.isnan(flat_data)))
    if precip:
        return [0, eps, 1.2, 2.2, 5.2, 8.7, 16.4, 26.9, 59.2, np.nanmax(data)]

    e = np.percentile(flat_data[:], np.arange(0, 100, nbins))
    if addZero:
        e = np.asarray([0, eps]+e.tolist()+[np.max(flat_data[:])])
    else:
        e = np.asarray(e.tolist()+[np.nanmax(flat_data[:])])
    return e

# Downloads OLR data from an online source for a specified year range.
def getOLR(start, end, onlinePath="https://www.ncei.noaa.gov/data/outgoing-longwave-radiation-daily/access/"):
    if not (os.path.exists('./olr/')):
        os.makedirs('olr', mode=0o777)
    for year in range(start, end+1):
        fname = f"olr-daily_v01r02_{year}0101_{year}1231.nc"
        if not (os.path.exists('./olr/'+fname)):
            print(">> ", fname)
            r = req.get(onlinePath+fname, allow_redirects=True)
            open(
                f"./olr/olr-daily_v01r02_{year}0101_{year}1231.nc", 'wb').write(r.content)

# Computes a histogram for data based on specified edges.
def getHist(data, edges):
    h, _ = np.histogram(data, bins=edges, density=True)
    return h

# Computes the symmetric Kullback-Leibler divergence between two distributions.
def getSDKL(d1, d2, eps=1e-7):
    d1 = d1 + eps*np.ones(len(d1))
    d2 = d2 + eps*np.ones(len(d2))
    return sum(sp.kl_div(d1, d2) + sp.kl_div(d2, d1))

# Computes the Earth Mover's Distance (ED) between two arrays, split into zones.
def edArray(d1, d2, nbz=4):
    d1p = np.array_split(d1, nbz)
    d2p = np.array_split(d2, nbz)
    m = 0
    for p in range(nbz):
        m += getSDKL(d1p[p], d2p[p])
    return m / nbz

# Formats an analysis matrix into a histogram matrix using masks and edges.
def formatHist(data, masks, edges, path="./", vname="", save=True):
    nbl, _ = data.shape
    nbm, _ = masks.shape
    out = []
    for l in range(nbl):
        line = []
        for m in range(nbm):
            res = data[l, :]*masks[m, :]
            res = np.delete(res, np.where(np.isnan(res)))
            line += getHist(res[:], edges).tolist()
        out.append(np.asarray(line))
    if save:
        if vname != "":
            if vname == "Vitesse":
                np.save(f"{path}histMatrix-{vname}", np.asarray(out))
            else:
                np.save(f"{path}histMatrix-{vname.upper()}", np.asarray(out))
        else:
            np.save(f"{path}histMatrix", np.asarray(out))
    else:
        return np.asarray(out)

# Removes NaN values from an analysis matrix.
def RemoveNan(AnalyseMatrix):
    nbligne, _ = AnalyseMatrix.shape
    new_mat = []
    for i in range(nbligne):
        new_mat.append(np.delete(AnalyseMatrix[i, :], np.where(
            np.isnan(AnalyseMatrix[i, :]))))
    return np.asarray(new_mat)

# Computes a distance matrix for the analysis data using either ED or Euclidean distance.
def distMatrix(data,ed=True, path="./", vname="",multi=False):
    if ed:
        if multi:
            tab = dist.cdist(data, data, edArrayMulti) 
        else:
            tab = dist.cdist(data, data, edArray)
        name = "ed"
    else:
        data = RemoveNan(data)
        tab = dist.cdist(data, data, metric="euclidean")
        name = "l2"

    np.save(f"{path+name}-dist-tab-{vname}", tab)

# Calculates wind speed from U and V components and saves the result.
def calculVitesse(u, v, path="./"):
    np.save(f"{path}AnalyseVitesse", np.sqrt(np.power(u, 2)+np.power(v, 2)))

# Calculates wind direction from U and V components and saves the result.
def calculDirection(u, v, path="./"):
    nbl, nbc = u.shape
    u = u.flatten()
    v = v.flatten()
    n = len(u)
    dir = np.empty(n)
    for i in range(n):
        if u[i] > 0 and v[i] > 0:
            dir[i] = ((3*np.pi)/2) + np.arctan(abs(v[i])/abs(u[i]))
        elif u[i] < 0 and v[i] > 0:
            dir[i] = np.arctan(abs(u[i])/abs(v[i]))
        elif u[i] < 0 and v[i] < 0:
            dir[i] = (np.pi/2)+np.arctan(abs(v[i])/abs(u[i]))
        elif u[i] > 0 and v[i] < 0:
            dir[i] = np.pi+np.arctan(abs(u[i])/abs(v[i]))
    dir = np.reshape(dir, (nbl, nbc))
    np.save(f"{path}AnalyseDirection", dir)
    return dir

# Concatenates multiple datasets along the second axis, with optional saving.
def concatData(datas,nb_jour=None,path="./",name="",save=False):
    if(nb_jour!=None):
        for d in datas:
            d=d[:nb_jour,:]
    concat=np.concatenate(datas,axis=1)
    if save:
        np.save(f"{path}{name}",concat)
    return concat

# Main function for generating analysis matrices, masks, and distance matrices.
def main(url, param, DataType, nbmasks, nanVal, add0=False):
    if not (os.path.exists('./MatricesAnalyse')):
        os.mkdir('./MatricesAnalyse/')

    print("# Génération de la matrice d'analyse")
    Alldays_matrix(url, param, DataType, coo=[-98.75, 31.25, -56.25, 8.75],
                   save=True, nanValue=nanVal, savepath='./MatricesAnalyse/')
    print('# Génération des masks')
    if param == "olr":
        CreateMasks(nbmasks, path="./masks/",
                    savepath="./MatricesAnalyse/", deg1=True)
    else:
        CreateMasks(nbmasks, path='./masks/', savepath='./MatricesAnalyse/')
    data = np.load(f'./MatricesAnalyse/Analyse{param.upper()}.npy')
    masks = np.load('./MatricesAnalyse/masks_etude.npy')
    if param == "precipitation":
        edges = getEdges(data, precip=True)
    else:
        edges = getEdges(data, addZero=add0)
    print("# Générations des matrices de distances")
    formatHist(data, masks, edges, path='./MatricesAnalyse/', vname=param)

# Performs Principal Component Analysis (PCA) on data and saves the transformed data.
def ACP(data,nb_compo,seuil=95,vname="",path="./MatricesAnalyse/"):
    acp=PCA(nb_compo)
    acp.fit(data)
    evr=acp.explained_variance_ratio_
    print("nb_compo pour "+vname,get_nb_comp(evr,seuil))
    acp=PCA(get_nb_comp(evr,seuil))
    new_data=acp.fit_transform(data)
    np.save(f"{path}AnalyseACP-{vname.upper()}",new_data)
    evr=evr=acp.explained_variance_ratio_
    return new_data,evr

# Determines the number of components needed to reach a specified variance threshold.
def get_nb_comp(evr,seuil=95):
    evr=np.cumsum(evr)
    i=np.where(evr >=95/100)
    if len(i[0])>0:
        return i[0][0]+1
    else:
        print(f"Le nombre de composante choisi n'est représente pas {seuil}% de vos données")
        return None

# Performs clustering on PCA-transformed data using K-Medoids and Agglomerative Clustering.
def clusterACP(data, kmax=15, path="./MatricesAnalyse/",vname="", save=True, load=False):
    print(">> clusterData")
    if load:
        if vname!="":
            print(f"{path}clustering-res-acp-{vname.upper()}.npy")
            res = np.load(f"{path}clustering-res-acp-{vname.upper()}.npy",allow_pickle=True).item()
        else:
            res = np.load(f"{path}clustering-res-acp.npy",allow_pickle=True).item()
        return res["kmed"],res["ahc"]
    
    kmed = {"l2":[]}
    ahc = {"l2":[]}
    l2tab=np.load(f"{path}l2-dist-tab-{vname}.npy")
    for k in range(2,kmax+1):
        if os.name=='nt' and ('win' in sys.platform):
            out=kmedoids.KMedoids(n_clusters=k,max_iter=10000,method='fasterpam').fit(l2tab)
            label=out.labels_
            kmed["l2"].append(label)
            print(f".kmed-l2 ({k-1}/{kmax-1})")
            out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', metric='precomputed').fit(l2tab)
            label=out.labels_
            ahc["l2"].append(label)
        else:
            out=skec.KMedoids(n_clusters=k,  max_iter=10000, init='k-medoids++').fit(data)
            label=out.labels_
            kmed["l2"].append(label)
            print(f".kmed-l2 ({k-1}/{kmax-1})")
            out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', affinity='precomputed').fit(l2tab)
            label=out.labels_
            ahc["l2"].append(label)
            print(f".ahc-l2 ({k-1}/{kmax-1})")
    kmed["l2"]=np.transpose(kmed["l2"])
    ahc["l2"]=np.transpose(ahc["l2"])

    if save:
        if vname!="":
            np.save(f"{path}clustering-res-acp-{vname.upper()}",{"kmed":kmed,"ahc":ahc})
        else:
            np.save(f"{path}clustering-res-acp",{"kmed":kmed,"ahc":ahc})
    return kmed, ahc

# Plots the explained variance ratio from PCA as a bar and line chart.
def plotACP(evr,seuil=95,path="./ACP/",fname="acp",title="Evolution of explained ratio variances according to the number of components",save=True,fs=16):
    y=[]
    evrcum=np.cumsum(evr)
    for e in evrcum:
        if e*100<=seuil:
            y.append(e*100)
        else:
            break
    y.append(evrcum[len(y)]*100)
    x=np.arange(len(y))
    
    fig,ax=plt.subplots(figsize=(12, 8))
    
    plt.grid()
   
    ax.bar(x,evr[:len(y)]*100,label="explained variance ratio")
    plt.xticks(np.arange(0,len(y),5))
    ax.set_xlabel('composantes')
    ax.set_ylabel('explained variance ratio')
    ax1=ax.twinx()
    ax1.plot(y,"r.-",label="cummulated explained variance ratio")
    
    ax1.set_ylim(0,100)
    ax1.axhline(seuil,color="green",label=f"seuil à {seuil}%")
    ax1.axhline(75,color="yellow",label=f"seuil à 75%")
    ax1.set_ylabel('cummulated explained variance ratio')
    plt.legend(loc="lower right")
    plt.title(f"{title}",fontsize=fs+2,fontweight="bold")
    
    if save:
        if not (os.path.exists('./ACP')):
            os.mkdir('./ACP/')
        plt.savefig(f"{path+fname}.png")
    plt.show

# Computes the Earth Mover's Distance (ED) for multi-dimensional data.
def edArrayMulti(d1,d2,nbz=4,nbp=3,moy=False):
    d1p=np.array_split(d1,nbp)
    d2p=np.array_split(d2,nbp)
    m=0
    for i in range(nbp):
        m+=edArray(d1p[i],d2p[i],nbz)
    if moy:
        return m/nbp
    else:
        return m

# Performs multi-parameter clustering using K-Medoids and Agglomerative Clustering.
def clusterMulti(data, kmax=15, path="./MatricesAnalyse/",vname="", save=True, load=False):
    print(">> clusterData")
    if load:
        if vname!="":
            print(f"{path}clustering-res-acp-{vname.upper()}.npy")
            res = np.load(f"{path}clustering-res-Multi-{vname.upper()}.npy",allow_pickle=True).item()
        else:
            res = np.load(f"{path}clustering-res-Multi.npy",allow_pickle=True).item()
        return res["kmed"],res["ahc"]
    
    kmed = {"ed":[],"l2":[]}
    ahc = {"ed":[],"l2":[]}
    print("l2tab : "+f"{path}l2-dist-tab-{vname}.npy")
    l2tab=np.load(f"{path}l2-dist-tab-{vname}.npy")
    print("histMat : "+f"./MatricesAnalyse/histMatrix-{vname.upper()}.npy")
    hist=np.load(f"./MatricesAnalyse/histMatrix-{vname.upper()}.npy")
    for k in range(2,kmax+1):
        print(f".kmed-ed ({k-1}/{kmax-1})")
        out=skec.KMedoids(n_clusters=k,  max_iter=10000, init='k-medoids++',metric=edArrayMulti).fit(hist)
        label=out.labels_
        kmed["ed"].append(label)
        print(f".kmed-l2 ({k-1}/{kmax-1})")
        out=skec.KMedoids(n_clusters=k,  max_iter=10000, init='k-medoids++').fit(data)
        label=out.labels_
        kmed["l2"].append(label)
        print(f".ahc-ed ({k-1}/{kmax-1})")
        out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', affinity=edArrayMulti ).fit(hist)
        label=out.labels_
        ahc["ed"].append(label)
        print(f".ahc-l2 ({k-1}/{kmax-1})")
        out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', affinity='precomputed').fit(l2tab)
        label=out.labels_
        ahc["l2"].append(label)
    kmed["l2"]=np.transpose(kmed["l2"])
    ahc["l2"]=np.transpose(ahc["l2"])

    if save:
        if vname!="":
            np.save(f"{path}clustering-res-multi-{vname.upper()}",{"kmed":kmed,"ahc":ahc}) # type: ignore
        else:
            np.save(f"{path}clustering-res-mutli",{"kmed":kmed,"ahc":ahc}) # type: ignore
    return kmed, ahc

# Performs multi-parameter clustering with precomputed distance matrices.
def clusterMulti2(data, kmax=15, path="./MatricesAnalyse/",vname="", save=True, load=False):
    print(">> clusterData")
    if load:
        if vname!="":
            print(f"{path}clustering-res-multi-{vname.upper()}.npy")
            res = np.load(f"{path}clustering-res-multi-{vname.upper()}.npy",allow_pickle=True).item()
        else:
            res = np.load(f"{path}clustering-res-multi.npy",allow_pickle=True).item()
        return res["kmed"],res["ahc"]
    kmed = {"ed":[],"l2":[]}
    ahc = {"ed":[],"l2":[]}
    print("l2tab : "+f"{path}l2-dist-tab-{vname}.npy")
    l2tab=np.load(f"{path}l2-dist-tab-{vname}.npy")
    print(f"ed-tab: {path}ed-dist-tab-{vname}.npy")
    edsumtab=np.load(f"{path}ed-dist-tab-{vname}.npy")
    for k in range(2,kmax+1):
        print(f".Kmed-l2 ({k-1}/{kmax-1})")
        out=kmedoids.KMedoids(n_clusters=k,max_iter=10000,method='fasterpam').fit(l2tab)
        label=out.labels_
        kmed["l2"].append(label)
        print(f".Kmed-ed ({k-1}/{kmax-1})")
        out=kmedoids.KMedoids(n_clusters=k,max_iter=10000,method='fasterpam').fit(edsumtab)
        label=out.labels_
        kmed["ed"].append(label)
        print(f".ahc-ed ({k-1}/{kmax-1})")
        out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', metric="precomputed" ).fit(edsumtab)
        label=out.labels_
        ahc["ed"].append(label)
        print(f".ahc-l2 ({k-1}/{kmax-1})")
        out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', metric='precomputed').fit(l2tab)
        label=out.labels_
        ahc["l2"].append(label)
    kmed["l2"]=np.transpose(kmed["l2"])
    ahc["l2"]=np.transpose(ahc["l2"])
    kmed["ed"]=np.transpose(kmed["ed"])
    ahc["ed"]=np.transpose(ahc["ed"])

    if save:
        if vname!="":
            np.save(f"{path}clustering-res-multi-{vname.upper()}",{"kmed":kmed,"ahc":ahc}) # type: ignore
        else:
            np.save(f"{path}clustering-res-mutli",{"kmed":kmed,"ahc":ahc}) # type: ignore
    return kmed, ahc

# Main function for clustering tasks, including distance matrix generation and multi-parameter clustering.
def main_cluster():# 1 Génération matrice distances # 2 Clustering uni-paramétrique # 3 Génération des matrices d'histogrammes # 4 Concaténation des données
    choix = int(sys.argv[1])
    
    if choix == 1:
        print("# Génération des matrices de distances")
        param = sys.argv[2]
        multi=bool(sys.argv[3])
        if param == "uv":
            matrice_hist = np.load(f'./MatricesAnalyse/histMatrix-Vitesse.npy')
            data = np.load(f'./MatricesAnalyse/AnalyseVitesse.npy')
        else:
            matrice_hist = np.load(f'./MatricesAnalyse/histMatrix-{param.upper()}.npy')
            data = np.load(f'./MatricesAnalyse/Analyse{param.upper()}.npy')
        print(f">> ed-dist-tab-{param}")
        distMatrix(matrice_hist,vname=param, path='./MatricesAnalyse/',multi=multi)
        
    elif choix == 2:
        print("# Clustering uni-paramétrique")
        k = int(sys.argv[3])
        vname = sys.argv[2]
        if vname == "uv":
            data = np.load(f'./MatricesAnalyse/histMatrix-Vitesse.npy')
        else:
            data = np.load(f"./MatricesAnalyse/histMatrix-{vname.upper()}.npy")

        clusterData(data, k, "./MatricesAnalyse/", vname)
    elif choix == 3:
        print("# Génération des matrices d'histogrammes")
        vname = sys.argv[2]
        nbmasks = int(sys.argv[3])
        add0 = bool(sys.argv[4])
        if vname=="Vitesse":
            data = np.load(f"./MatricesAnalyse/Analyse{vname}.npy") 
        else:
            data = np.load(f"./MatricesAnalyse/Analyse{vname.upper()}.npy")
        if vname == "olr":
            CreateMasks(nbmasks, path="./masks/", savepath="./MatricesAnalyse/", deg1=True)
        else:
            CreateMasks(nbmasks, path='./masks/',savepath='./MatricesAnalyse/')

        masks = np.load('./MatricesAnalyse/masks_etude.npy')
        if vname == "precipitation":
            edges = getEdges(data, precip=True)
        else:
            edges = getEdges(data, addZero=add0)
        
        formatHist(data, masks, edges, path='./MatricesAnalyse/', vname=vname)
    elif choix==4:
        print("# Concaténation des données")
        nbj=int(sys.argv[-1])
        vars=sys.argv[2:len(sys.argv)-1]
        datas=[]
        name=""
        for vname in vars:
            if vname=="Vitesse":
                datas.append(np.load(f"./MatricesAnalyse/Analyse{vname}.npy")) 
            else:
                datas.append(np.load(f"./MatricesAnalyse/Analyse{vname.upper()}.npy"))
            if "ACP" in vname.upper():
                name+="-"+vname.split("-")[1]
            else:
                name+="-"+vname
            name+="-"+vname
        concatData(datas,nbj,"./MatricesAnalyse/",name)
    elif choix==5:
        print("# ACP")
        vname = sys.argv[2]
        seuil=int(sys.argv[3])
        anneefin=int(sys.argv[4].split("-")[2])
        nbj=getIdByDate(sys.argv[4],1979,anneefin)
        if vname=="Vitesse":
            data=np.load(f"./MatricesAnalyse/Analyse{vname}.npy")[:nbj+1,:] 
        else:
            data = np.load(f"./MatricesAnalyse/Analyse{vname.upper()}.npy")[:nbj+1,:]
        data=RemoveNan(data)
        print("data.shape= ",data.shape)
        ACP(data,None,seuil,vname)
    elif choix==6: 
        print("# Clustering multi-paramétrique")
        kmax=int(sys.argv[-1])
        
        vars=sys.argv[2:len(sys.argv)-1]
        datas=[]
        name=""
        print("# Concaténation des données")
        for vname in vars:
            datas.append(np.load(f"./MatricesAnalyse/Analyse{vname.upper()}.npy"))
            if "ACP" in vname.upper():
                name+="-"+vname.split("-")[1]
            else:
                name+="-"+vname
        concatData(datas,15706,"./MatricesAnalyse/",name)
        data=np.load(f"./MatricesAnalyse/Analyse-concat{name.upper()}.npy")
        print("data.shape= ",data.shape)
        print("# Génération matrices distance")
        distMatrix(data,ed=False,path="./MatricesAnalyse/",vname=name[1:])
        print("# Clustering ")
        clusterACP(data,vname=name[1:],kmax=kmax)
    elif choix ==7:
        if not(os.path.exists("./silhouettes")):
            os.makedirs('silhouettes', mode=0o777)
        vname=sys.argv[2]
        fname=sys.argv[3]
        vmin=float(sys.argv[4])
        vmax=float(sys.argv[5])
        kmed,ahc=clusterACP(None,load=True,vname=vname)
        getSilhouette(kmed["l2"],kmed["l2"],mth="KMED-l2",path="./MatricesAnalyse/",vname=vname)
        getSilhouette(ahc["l2"],ahc["l2"],mth="AHC-l2",save=True,style="g.-",fname=fname,ylim=(vmin,vmax),path="./MatricesAnalyse/",savepath="./silhouettes/",vname=vname)
    elif choix==8:
        start = time.time()
        print("start")
        data=np.load("./MatricesAnalyse/Analyse-concat-SST-VITESSE-OLR.npy")
        print("data shape",data.shape)
        kmed,ahc=clusterMulti(data,vname="sst-vitesse-olr")
        hist=np.load(f"./MatricesAnalyse/histMatrix-SST-VITESSE-OLR.npy")
        l2tab=np.load(f"./MatricesAnalyse/l2-dist-tab-sst-vitesse-olr.npy")
        sil=[]
        for k in range(2,15+1):
            sil.append(float(skm.silhouette_score(hist,kmed["ed"][:,(k-2)],metric=edArrayMulti)))
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.plot(sil,"r.-", linewidth=2, markersize=18, label="kmed-ed".upper())

        sil=[]
        for k in range(2,15+1):
            sil.append(float(skm.silhouette_score(data,kmed["l2"][:,(k-2)])))
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.plot(sil,"k.-", linewidth=2, markersize=18, label="kmed-l2".upper())


        sil=[]
        for k in range(2,15+1):
            sil.append(float(skm.silhouette_score(hist,ahc["ed"][:,(k-2)],metric=edArrayMulti)))
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.plot(sil,"r.--", linewidth=2, markersize=18, label="ahc-ed".upper())

        sil=[]
        for k in range(2,15+1):
            sil.append(float(skm.silhouette_score(l2tab,ahc["l2"][:,(k-2)],metric="precomputed")))
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.plot(sil,"k.--", linewidth=2, markersize=18, label="ahc-l2".upper())


        plt.xlabel('Number of clusters (k)',fontsize=18)
        plt.ylabel('Silhouette index',fontsize=18)
        plt.legend(fontsize=18-2)
        plt.xlim(0,15)
        plt.ylim(-1,1)
        plt.grid()
        plt.title("Silhouette indexes")
        now = dt.datetime.today()
        date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
        plt.savefig("./silhouttes"+"sil_"+"sst-vitesse-olr-edm"+"_" + date_time+".png",dpi=600)
        end = time.time()
        elapsed = end - start
        print(f'Temps d\'exécution : {elapsed/60}')
    elif choix==9:
        start = time.time()
        print("start")
        kmed,ahc=clusterMulti2(None,vname="sst-vitesse-olr")
        print("#Fin clustering")
        getSilhouette(kmed["ed"],kmed["ed"],ed=True,mth="KMED-ED",style="r.-",path="./MatricesAnalyse/",vname="sst-vitesse-olr")
        getSilhouette(kmed["l2"],kmed["l2"],mth="KMED-l2",path="./MatricesAnalyse/",vname="sst-vitesse-olr")

        getSilhouette(ahc["ed"],ahc["ed"],ed=True,mth="AHC-ED",style="r.--",path="./MatricesAnalyse/",vname="sst-vitesse-olr")
        getSilhouette(ahc["l2"],ahc["l2"],mth="AHC-l2",save=True,style="k.--",fname=f"AHC-KMED-sst-vitesse-olr-edm",ylim=(0,1),path="./MatricesAnalyse/",savepath="./silhouettes/",vname="sst-vitesse-olr")
        end = time.time()
        elapsed = end - start
        print(f'Temps d\'exécution : {elapsed/60}')

    else:
        print("Valeur incorrecte.\n")
        print("# 1 Génération des matrices de  distances")
        print(f"Usage: python {sys.argv[0]} 1 vname multi(True/False)")
        print("# 2 Clustering uni-paramétrique")
        print(f"Usage: python {sys.argv[0]} 2 vname nb_cluster")
        print("# 3 Génération des matrices d'histogrammes")
        print(f" Usage: python {sys.argv[0]} 3 vname nb_masks add0(True/False)")
        print("# 4 Concaténation des données")
        print(f"Usage: python {sys.argv[0]} 4 vname_1 vname_2 vname_n nombrejour")
        print("# 5 ACP")
        print(f"Usage: python {sys.argv[0]} 5 vname seuil date_fin(jj-mm-aaaa)")
        print("# 6 Clustering multi")
        print(f"Usage: python {sys.argv[0]} 6 vname_1 vname_2 vname_n nb_jour kmax")
        print("# 7 Silhouette")
        print(f"Usage: python {sys.argv[0]} 7 vname fname ymin ymax")

# Retrieves cluster information, including cluster centers and representative days.
def getClusterInfoMulti(analyse, idx, ed=False):
    info = {"cc":[],"n":[],"rc":[]}
    n,_ = analyse.shape
    k = int(np.max(idx) + 1)
    for c in range(k):
        sel = idx==c
        info["n"].append(np.sum(sel))
        l=[]
        for d in range(n):
            if idx[d]==c:
                l.append(analyse[d,:])
        l = np.asarray(l)
        info["cc"].append(np.sum(l,axis=0)/info["n"][c])
        for d in range(n):
            if ed:
                dist = edArrayMulti(info["cc"][c], analyse[d,:])
            else:
                dist = np.linalg.norm(info["cc"][c] - analyse[d,:])
            
            if d == 0:
                info["rc"].append(d)
                dmin = dist
            elif dist < dmin and idx[d]==c:
                info["rc"][c] = d
                dmin = dist
    return info

# Returns a list of the top N representative days for each cluster.
def getClusterRClist(analyse, idx,nb_rc=5, ed=False):
    rcList=[]
    n,_ = analyse.shape
    k = int(np.max(idx) + 1)
    for c in range(k):
        rcList.append([])
        for d in range(n):
            if ed:
                dist = edArrayMulti(info["cc"][c], analyse[d,:])
            else:
                dist = np.linalg.norm(info["cc"][c] - analyse[d,:])
            if idx[d]==c:
                rcList[c].append((d,dist))
        rcList[c]=sorted(rcList[c],key=lambda row: row[1])
    return rcList