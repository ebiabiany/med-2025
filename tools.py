import os
import sys
#import scipy.ndimage as scni
#import scipy.stats as scs
import numpy as np
import datetime as dt
import scipy.special as sp

import sklearn.cluster as skc
import sklearn.metrics as skm
import sklearn.neighbors as skn
import matplotlib.pyplot as plt
import sklearn_extra.cluster as skec
if os.name=='nt' and ('win' in sys.platform):
    import netCDF4 as nc  
    from mpl_toolkits.basemap import Basemap
    from multiprocessing import Pool
    import pymannkendall as mk
from scipy.spatial import distance as dist
from scipy.sparse import csgraph 
#from mpl_toolkits.basemap import maskoceans


#Include 0 bin for rainfall data
def getEdges(data, nbins=10, eps=1e-7):
    #_,e = np.histogram(data,bins=nbins)
    e = np.percentile(data, np.arange(0, 100, nbins))
    if e[0]==0:
        e = np.asarray([0,eps]+e[1:].tolist()+[np.max(data)])
    else:
        e = np.asarray([0,eps]+e.tolist()+[np.max(data)])
    #print(".getEdges "+str(e))
    return e

#Procudes histogram based on selected edges
def getHist(data,edges):
    h,_ = np.histogram(data,bins=edges,density=True)
    return h

#Produces histogram analyse matrix
def formatHist(data, edges):
    nbl,_ = data.shape
    out = []
    for l in range(nbl):
        out.append(getHist(data[l,:],edges))
    return np.asarray(out)

#Produces symetried kullback-leibler divergency
def getSDKL(d1,d2,eps=1e-7):
    d1 = d1 + eps*np.ones(len(d1))
    d2 = d2 + eps*np.ones(len(d2))
    return sum(sp.kl_div(d1,d2) + sp.kl_div(d2,d1))

#Produces ed metric from list
def edList(d1,d2):
    m = 0
    for p in range(len(d1)):
        m += getSDKL(d1[p],d2[p])
    return m / len(d1)

#Produces ed metric from array
def edArray(d1,d2,nbp=4):
    d1p = np.array_split(d1,nbp)
    d2p = np.array_split(d2,nbp)
    m = 0
    for p in range(nbp):
        m += getSDKL(d1p[p],d2p[p])
    return m / nbp

#Produce ED distance matrix
def getEDDistTab(data, path=".",save=True, load=False):
    print(".getEDDistTab")
    if load:
        return np.load(path+"ed-dist-tab.npy")
    n,_ = data.shape
    tab = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            tab[i,j] = edArray(data[i,:],data[j,:])
            tab[j,i] = tab[i,j]
    if save:
        np.save(path+"ed-dist-tab",tab)
    return tab

#Produce ED distance matrix
# def getEDDistTabPar(data, path=".",save=True, load=False):
#     print(".getEDDistTabPar")
#     if load:
#         return np.load(path+"ed-dist-tab.npy")
#     def f(d):
#         return edArray(data[d[0],:],data[d[1],:])
#     n,_ = data.shape
#     d=[]
#     for i in range(n-1):
#         for j in range(i+1,n):
#             d.append([i,j])
#     with Pool(6) as p:
#         res=p.map(f,d)
#     tab = np.zeros((n,n))
#     for i in range(len(d)):
#         tab[d[i][0],d[i][1]] = res[i]
#         tab[d[i][1],d[i][0]]
#     if save:
#         np.save(path+"ed-dist-tab",tab)
#     return tab

# def storeDistMatrix(data,ed=False,path="."):
#     if ed :
#         tab = dist.cdist(data,data,metric=edArray)
#         fname = "ed"
#     else:
#         tab = dist.cdist(data,data,metric="euclidean")
#         fname = "l2"
#     print(".storeDistMatrix-"+fname)
#     np.save(path+fname+"-dist-tab",tab)


#Produces clustering KMED, AHC
def clusterData(data, kmax=15, path=".",vname="", save=True, load=False):
    print(">> clusterData")
    if load:
        if vname!="":
            print(f"{path}clustering-res-{vname}.npy")
            res = np.load(f"{path}clustering-res-{vname}.npy",allow_pickle=True).item()
        else:
            res = np.load(f"{path}clustering-res.npy",allow_pickle=True).item()
        return res["kmed"],res["ahc"],res["spec"],res["dbscan"]
    kmed = {"ed":[],"l2":[]}
    ahc = {"ed":[],"l2":[]}
    dbscan={"ed":[],"l2":[]}
    # fa={"ed":[],"l2":[]} pareil que ahc
    spec={"ed":[],"l2":[]}
    if vname!="":
        edtab=np.load(f"{path}ed-dist-tab-{vname}.npy")
        l2tab=np.load(f"{path}l2-dist-tab-{vname}.npy")
    else:
        edtab=np.load(f"{path}ed-dist-tab.npy")
        l2tab=np.load(f"{path}l2-dist-tab.npy")
    sparse_graph_ed=skn.sort_graph_by_row_values(csgraph.csgraph_from_dense(edtab))
    sparse_graph_l2=skn.sort_graph_by_row_values(csgraph.csgraph_from_dense(l2tab))
    for k in range(2,kmax+1):
        out=skec.KMedoids(n_clusters=k,  max_iter=10000, metric=edArray, init='k-medoids++').fit(data) # type: ignore
        label=out.labels_
        kmed["ed"].append(label)
        print(f".kmed-ed ({k-1}/{kmax-1})")
        out=skec.KMedoids(n_clusters=k,  max_iter=10000, init='k-medoids++').fit(data)
        label=out.labels_
        kmed["l2"].append(label)
        print(f".kmed-l2 ({k-1}/{kmax-1})")

        out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', metric='precomputed').fit(edtab)
        label=out.labels_
        ahc["ed"].append(label)
        print(f".ahc-ed ({k-1}/{kmax-1})")
        out=skc.AgglomerativeClustering(n_clusters=k, linkage='average', metric='precomputed').fit(l2tab)
        label=out.labels_
        ahc["l2"].append(label)        
        print(f".ahc-l2 ({k-1}/{kmax-1})")
        # ----------------------------------------------
        out=skc.SpectralClustering(n_clusters=k,affinity="precomputed_nearest_neighbors").fit(sparse_graph_ed)
        label=out.labels_
        spec["ed"].append(label)
        print(f".spec-ed ({k-1}/{kmax-1})")
        out=skc.SpectralClustering(n_clusters=k,affinity="precomputed_nearest_neighbors").fit(sparse_graph_l2)
        label=out.labels_
        spec["l2"].append(label)        
        print(f".spec-l2 ({k-1}/{kmax-1})")
        # ----------------------------------------------
        # out=skc.FeatureAgglomeration(n_clusters=k,metric="precomputed",linkage="average",pooling_func=np.average).fit(edtab)
        # label=out.labels_
        # fa["ed"].append(label)
        # print(f".FA-ed ({k-1}/{kmax-1})")
        # out=skc.FeatureAgglomeration(n_clusters=k, linkage='average', metric='precomputed',pooling_func=np.average).fit(l2tab)
        # label=out.labels_
        # fa["l2"].append(label)        
        # print(f".FA-l2 ({k-1}/{kmax-1})")
# -------------------------------------------
    out=skc.DBSCAN(metric='precomputed',min_samples=5,eps=7800).fit(edtab)
    label=out.labels_
    dbscan["ed"].append(label)
    out=skc.DBSCAN(metric='precomputed',min_samples=5,eps=2000).fit(l2tab)
    label=out.labels_
    dbscan["l2"].append(label)
    # out=skec.CommonNNClustering(metric="precomputed").fit(edtab)
    # label=out.labels_
    # common["ed"].append(label)
    # out=skec.CommonNNClustering(metric='precomputed').fit(l2tab)
    # label=out.labels_
    # common["l2"].append(label)
#------------------------------------------------ 
    kmed["ed"]=np.transpose(kmed["ed"])# type: ignore
    kmed["l2"]=np.transpose(kmed["l2"])# type: ignore
    ahc["ed"]=np.transpose(ahc["ed"])# type: ignore
    ahc["l2"]=np.transpose(ahc["l2"])# type: ignore
    spec["ed"]=np.transpose(spec["ed"])# type: ignore
    spec["l2"]=np.transpose(spec["l2"])# type: ignore
    dbscan["l2"]=np.transpose(dbscan["l2"])# type: ignore
    dbscan["ed"]=np.transpose(dbscan["ed"]) # type: ignore
    # common["l2"]=np.transpose(common["l2"])# type: ignore
    # common["ed"]=np.transpose(common["ed"]) # type: ignore
    # fa["l2"]=np.transpose(fa["l2"])# type: ignore
    # fa["ed"]=np.transpose(fa["ed"]) # type: ignore
    if save:
        if vname!="":
            np.save(f"{path}clustering-res-{vname}",{"kmed":kmed,"ahc":ahc,"spec":spec,"dbscan":dbscan}) # type: ignore
        else:
            np.save(f"{path}clustering-res",{"kmed":kmed,"ahc":ahc,"spec":spec,"dbscan":dbscan}) # type: ignore
    return kmed, ahc,spec,dbscan

#Load a formated roc csvfile
def loadRCFile(filename,path=""):
    print(".load : "+filename)
    data = np.loadtxt(path+filename, delimiter=";")
    data = data[1:,1:]
    return data.astype(float), filename.replace(".csv","")

#Load a formated roc csvfile
def loadRCFile2(filename,path=""):
    print(".load : "+filename)
    data = np.loadtxt(path+filename, delimiter=",",dtype=str)
    data = data[3:,:]
    #data = np.nan_to_num(data,nan=0.0)
    return data.astype(float), filename.replace(".csv","")

#Produces the analyse matrix
def loadAnalyse(path=".",save=True, load=False,version=1):
    print(">> loadAnalyse : "+path)
    if load:
        return np.load(path+"analyse.npy",allow_pickle=True).item()
    data = []
    label = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            if version==1:
                d,l = loadRCFile(file,path)
            else:
                d,l = loadRCFile2(file,path)
            data.append(d)
            label.append(l)
    data_flat = [item for sublist in data for zone in sublist for item in zone.tolist()]
    edges = getEdges(data_flat)
    print(".formatData ")
    for d in range(len(data)):
        if d == 0:
            analyseA = formatHist(data[d],edges)
            analyseB = data[d]
        else:
            analyseA = np.concatenate((analyseA,formatHist(data[d],edges)),axis=1)
            analyseB = np.concatenate((analyseB,data[d]), axis=1)
    if save:
        print(".save analyse.npy ")
        np.save(path+"analyse",{"ed":analyseA,"l2":analyseB})
    return {"ed":analyseA,"l2":analyseB}

#Produces silhouette index values and graphs
def getSilhouette(analyse,idx,fs=18,lw=2,ms=10,kmax=15,style='k.-',ed=False, path="./",savepath="./",mth="", save=False,vname="",ftitle="Silhouette indexes",fname="",dpi=600,ylim=(-1,1)):
    print(".getSilhouette "+mth)
    sil = [float("nan"),float("nan")]
    plt.rcParams.update({'font.size': fs})
    if vname!="":
        if ed:
            print(f"{path}ed-dist-tab-{vname}.npy")
            tab=np.load(f"{path}ed-dist-tab-{vname}.npy")
        else:
            print(f"{path}l2-dist-tab-{vname}.npy")
            tab=np.load(f"{path}l2-dist-tab-{vname}.npy")
    else:
        if ed:
            tab=np.load(f"{path}ed-dist-tab.npy")
        else:
            tab=np.load(f"{path}l2-dist-tab.npy")
    if kmax==0:
        _,kmax=idx.shape
        kmax+=2
    for k in range(2,kmax+1):
        sil.append(float(skm.silhouette_score(tab,idx[:,(k-2)],metric="precomputed")))
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.plot(sil, style, linewidth=lw, markersize=ms, label=mth.upper())
    if save:
        plt.xlabel('Number of clusters (k)',fontsize=fs)
        plt.ylabel('Silhouette index',fontsize=fs)
        plt.legend(fontsize=fs-2)
        plt.xlim(0,kmax)
        plt.ylim(ylim)
        plt.grid()
        plt.title(ftitle)
        now = dt.datetime.today()
        date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
        plt.savefig(savepath+"sil_"+fname+"_" + date_time+".png",dpi=dpi)
        # plt.show()

def getCalHar(analyse,idx,fs=16,lw=2,ms=10,kmax=15,style='k.-',ed=False, path=".",mth="", save=False, ftitle="Calinski-Harabasz indexes",fname="",dpi=600):
    print(".getCalHar "+mth)
    sil = [float("nan"), float("nan")]
    if kmax==0:
        _,kmax=idx.shape
        kmax+=2
    for k in range(2,kmax+1):
        if ed:
            sil.append(calharED(analyse,idx[:,(k-2)]))
        else:
            sil.append(skm.calinski_harabasz_score(analyse,idx[:,(k-2)]))
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.plot(sil, style, linewidth=lw, markersize=ms, label=mth.upper())
    if save:
        plt.xlabel('Number of clusters (k)',fontsize=fs)
        plt.ylabel('Calinski-Harabasz index',fontsize=fs)
        plt.legend(fontsize=fs-2)
        plt.xlim(0,kmax)
        plt.grid()
        plt.title(ftitle)
        now = dt.datetime.today()
        date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
        plt.savefig(path+"calhar_"+fname+"_" + date_time+".png",dpi=dpi)
        plt.show()

#Produces information about clusters
def getClusterInfo(analyse, idx, ed=False):
    #print(">> getClusterInfo")
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
        #print(l.shape)
        info["cc"].append(np.sum(l,axis=0)/info["n"][c])
        #print("cc ", info["cc"][c].shape)
        for d in range(n):
            if ed:
                dist = edArray(info["cc"][c], analyse[d,:])
            else:
                dist = np.linalg.norm(info["cc"][c] - analyse[d,:])
            
            if d == 0:
                info["rc"].append(d)
                dmin = dist
            elif dist < dmin and idx[d]==c:
                info["rc"][c] = d
                dmin = dist
    return info

#Produce Calinski-Harabazs index usin ED
def calharED(X,label,info=False):
    n,_ = X.shape
    k = int(np.max(label) + 1)
    if not(info):
        info = getClusterInfo(X, label, ed=True)
    center = np.mean(X,axis=0)
    B = 0
    W = []
    for c in range(k):
        B += info["n"][c]*edArray(info["cc"][c],center)
        w = 0
        for d in range(n):
            if label[d]==c:
                w += edArray(X[d,:],info["cc"][c])
        
        W.append(w)
    return ((n-k)*B)/((k-1)*sum(W))

#Produces lats lons object for each zone
def loadPoints(path='.', version=1):
    print(">> loadPoints : "+path)
    pts = []
    i=0
    for file in os.listdir(path):
        if file.endswith(".csv"):
            print(".load : "+file)
            if version==1:
                data = np.loadtxt(path+file, delimiter=";")
            else:
                data = np.loadtxt(path+file, delimiter=",", dtype=str)
                data = data[1:3,:]
            data = data.astype(float)
            pts.append({"lons":data[0,:], "lats":data[1,:]})
            if i==0:
                all = data[0:2,:]
            else:
                all = np.concatenate((all,data[0:2,:]),axis=1)
            i += 1
    return pts, {"lons":all[0,:], "lats":all[1,:]}

def reMesh(data,lats,lons,lats_,lons_):
    out = np.ones((len(lats_),len(lons_)))*float('nan')
    for p in range(len(data)):
        i = np.argmin(np.absolute(lats_-lats[p]))
        j = np.argmin(np.absolute(lons_-lons[p]))
        #print(i,j)
        out[i,j] = data[p]
    return out

def getMesh(all):
    res=abs(all["lons"][0]-all["lons"][1])
    lats_min,lats_max = (np.min(all["lats"]), np.max(all["lats"]))
    lons_min,lons_max = (np.min(all["lons"]), np.max(all["lons"]))
    #print("lats_min => ",lats_min," lats_max => ",lats_max)
    #print("lons_min => ",lons_min," lons_max => ",lons_max)
    #print("res => ",res)
    #exit(0)
    lats_ = np.arange(lats_min,lats_max+res,res)
    lons_ = np.arange(lons_min,lons_max+res,res)
    xx, yy = np.meshgrid(lons_, np.flip(lats_))
    return lats_, lons_, xx, yy

# def plotPtsLoc(pts,box=[-80.,22.,-66.,16.], res="i", proj="merc",fname="exemple",dpi=600,fs=16):
#     # create new figure, axes instances.
#     plt.rcParams.update({'font.size': fs})
#     fig=plt.figure(figsize=((8,6)))

#     # setup mercator map projection #-66.,17.,-55.,8.
#     m = Basemap(llcrnrlon=box[0],llcrnrlat=box[3],urcrnrlon=box[2],urcrnrlat=box[1],resolution=res,projection=proj)

#     # print coastlines
#     m.drawcoastlines()
#     #m.fillcontinents()

#     # draw parallels and meridians
#     m.drawparallels(np.arange(-90,90,5),labels=[1,1,1,1])
#     m.drawmeridians(np.arange(-180,180,5),labels=[1,1,1,1])

#     for p in pts:
#         x, y = m(p["lons"], p["lats"])
#         m.plot(x, y, '.')

#     plt.xlabel("\nLongitude (°W)")
#     plt.ylabel("Latitude (°N)")
#     plt.show()

def testBissextile(annee):
    log = annee%400==0 or annee%4==0
    if log :
        jr=366
    else :
        jr=365
    return [log,jr]

def listDatesAnnee(annee,bic=None):
  dates = []
  nbj_m = [31,28,31,30,31,30,31,31,30,31,30,31]
  if bic==True:
    nbj_m[1] = 29
  for mois in range(1,12+1):
    for jour in range (1,nbj_m[mois-1]+1):
      dates.append([jour,mois,annee])
  return dates

def getMonthById(id,startYear):
    m = (id%12) + 1
    a = startYear+int(id/12)
    return m,a

def date2str(date,option=None):
    if date[0]<10:
        dd = '0'+str(date[0])
    else:
        dd = str(date[0])

    if date[1]<10:
        mm = '0'+str(date[1])
    else:
        mm = str(date[1])

    yyyy = str(date[2])

    mois = ['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aout','Sept','Oct','Nov','Dec']

    if option==None:
        return [dd,mm,yyyy]
    return {
        '/':dd+'/'+mm+'/'+yyyy,
        '-':dd+'-'+mm+'-'+yyyy,
        'inv':yyyy+mm+dd,
        '/inv':yyyy+'/'+mm+'/'+dd,
        '-inv':yyyy+'-'+mm+'-'+dd,
        'us':mm+dd+yyyy,
        '/us':mm+'/'+dd+'/'+yyyy,
        '-us':mm+'-'+dd+'-'+yyyy,
        'MM':str(date[0])+' '+mois[date[1]-1]+' '+yyyy
    }.get(option)

def getDateById(id,annee_debut,annee_fin,bic=None):
  dates = []
  if bic==None or bic==True:
      for annee in range(annee_debut,annee_fin+1):
        t,_ = testBissextile(annee)
        d = listDatesAnnee(annee,bic=t)
        dates.extend(d)
  else :
    for annee in range(annee_debut,annee_fin+1):
      d = listDatesAnnee(annee)
      dates.extend(d)

  f1 = date2str(dates[id-1],'-')
  f2 = date2str(dates[id-1],'MM')
  f3 = dates[id-1]

  return f1,f2,f3

def getInterAnnual(idx,mth="",path=".",dpi=600, ms=10, lw=2, fs=16, day=False,lim=.03):
    k = int(np.max(idx)+1)
    plt.rcParams.update({'font.size': fs})
    for c in range(k):
        x = []
        n = 0
        for d in range(len(idx)):
            if idx[d] == c:
                if day:
                    _,_,a = getDateById(d+1,1979,2021)
                    a = a[2]
                else:
                    _,a = getMonthById(d,1979)
                x.append(a)
                n += 1
        plt.figure(figsize=((10,6)))
        h,_ = np.histogram(x,bins=(max(x)-min(x)+1))
        h = h / len(x)
        m = np.arange(min(x),max(x)+1)
        coef = np.polyfit(m,h,1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(m, h, ".-k", linewidth=lw, markersize=ms, label="data")
        if coef[0]<0 and abs(coef[0])>1e-7:
            style="--r"
        elif coef[0]>0 and abs(coef[0])>1e-7:
            style="--g"
        else:
            style="--b"
        t, _, pv, _, _, _, _, _, _ = mk.original_test(h)
        print(f">> {mth}-K{k}-C{c+1} ({n})")
        plt.plot(m, poly1d_fn(m), style, linewidth=lw, markersize=ms, label="linear reg. ("+ r'$\alpha$'+f":{coef[0]:.2};p-value:{pv:.2}:{t})")
        plt.xlabel("Years")
        plt.ylabel("Frequency (%)")
        plt.ylim(0,lim)
        plt.legend()
        plt.title(f"{mth}-K{k}-C{c+1} ({n})",fontsize=fs+2,fontweight="bold")
        plt.savefig(f"{path}inter_{mth.lower()}-k{k}-c{c+1}.png",dpi=dpi)

def getIntraAnnual(idx,anneeDep,anneeFin,mth="",path=".",vname="",dpi=600, fs=16, day=False, lim=.15):
    k = int(np.max(idx)+1)
    plt.rcParams.update({'font.size': fs})
    colors = ["#C0392B","#2980B9","#27AE60","#F1C40F","orange","purple","blue","red","darkorchid","midnightblue"]
    for c in range(k):
        x = []
        n = 0
        for d in range(len(idx)):
            if idx[d] == c:
                if day:
                    _,_,m=getDateById(d+1,anneeDep,anneeFin)
                    
                    m = m[1]
                else:
                    m,_ = getMonthById(d,anneeDep)
                x.append(m)
                n += 1
        print(f">> {mth}-K{k}-C{c+1} ({n})")
        plt.figure(figsize=((10,6)))
        h,_ = np.histogram(x,bins=12)
        h = h / len(idx)
        m = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        plt.bar(m, h, color=colors[c])
        plt.xlabel("Months")
        plt.ylabel("Frequency (%)")
        plt.ylim(0,lim)
        plt.title(f"{mth}-K{k}-C{c+1} ({n})",fontsize=fs+2,fontweight="bold")
        plt.savefig(f"{path}intra_{mth.lower()}-k{k}-c{c+1}{vname}.png",dpi=dpi)
        plt.close()

def plotData(data,xx,yy,box=[-75.,20.2,-71.,17.9], res="i", proj="merc",fname="exemple",dpi=600,fs=18,path="./",title="exemple",cmax=550):
    # create new figure, axes instances.
    plt.rcParams.update({'font.size': fs})
    fig=plt.figure(figsize=((10,6)))

    # setup mercator map projection #-66.,17.,-55.,8.
    m = Basemap(llcrnrlon=box[0],llcrnrlat=box[3],urcrnrlon=box[2],urcrnrlat=box[1],resolution=res,projection=proj)

    # print coastlines
    m.drawcoastlines()
    #m.fillcontinents()

    # draw parallels and meridians
    m.drawparallels(np.arange(-90,90,10),labels=[1,0,0,1])
    m.drawmeridians(np.arange(-180,180,10),labels=[1,0,0,1])
    
    x,y = m(xx,yy)
    # mdata = maskoceans(x, y, np.flip(data,axis=0), resolution = 'i', grid = 1.25, inlands=True)

    # m.pcolormesh(x, y,data)
    #levels = np.linspace(0, cmax, 30)
    m.contourf(x, y, data, 20)
    #m.contourf(x, y, np.flip(data,axis=0), levels, cmap='Blues',)


    plt.xlabel("\nLongitude (°W)")
    plt.ylabel("Latitude (°N)\n\n")
    plt.colorbar()
    # plt.clim(0,cmax)
    plt.title(f"{title}",fontsize=fs+2,fontweight="bold")
    #plt.show()
    plt.savefig(f"{path}{fname}.png",dpi=dpi)

# #Main function
# def main():

#     url = "./data/mensuel/decoupe2/"
#     analyse = loadAnalyse(url, load=True, version=2)
#     print(np.max(analyse["l2"].flatten()))
#     exit(0)
#     #storeDistMatrix(analyse["l2"],path=url)
#     #storeDistMatrix(analyse["ed"],path=url,ed=True)
#     kmed,ahc = clusterData(analyse, path=url, load=True)
#     np.savetxt(url+"/res/idx_kmed-ed.csv",kmed["ed"],fmt='%d',delimiter=";")
#     l,_=kmed["ed"].shape
#     dates=[]
#     for i in range(l):
#         m,a =  getMonthById(i,1981)
#         dates.append(f"{m:02}-{a}")
#     np.savetxt(url+"/res/dates.csv",dates,fmt='%s',delimiter=";")
#     exit(0)
    
#     """
#     #--
#     plt.figure(figsize=(8,6))
#     plt.rcParams.update({'font.size': 16})
#     getSilhouette(analyse["ed"], kmed["ed"], ed=True, path=url, mth="kmed-ed", style="r.-")
#     #getSilhouette(analyse["ed"], ahc["ed"], ed=True, path=url, mth="ahc-ed", style="r.--")
#     getSilhouette(analyse["l2"], kmed["l2"], path=url, mth="kmed-l2", style="k.-", save=True, fname="all")
#     #getSilhouette(analyse["l2"], ahc["l2"], path=url, mth="ahc-l2", style="k.--", save=True, fname="all")
#     plt.close('all')
#     #---
    
#     plt.figure(figsize=(8,6))
#     plt.rcParams.update({'font.size': 16})
#     getCalHar(analyse["ed"], kmed["ed"], ed=True, path=url, mth="kmed-ed", style="b.-")
#     getCalHar(analyse["ed"], ahc["ed"], ed=True, path=url, mth="ahc-ed", style="b.--")
#     getCalHar(analyse["l2"], kmed["l2"], path=url, mth="kmed-l2", style="k.-")
#     getCalHar(analyse["l2"], ahc["l2"], path=url, mth="ahc-l2", style="k.--", save=True, fname="all")
#     plt.close('all')
#     #--

#     #"""

#     #k=int(input("?> Choose the number of cluster\n>> "))
#     k=7
#     info = getClusterInfo(analyse["ed"], kmed["ed"][:,k-2], ed=True)
#     #print(calharED(analyse["ed"], kmed["ed"][:,2]))
#     pts,all=loadPoints(url,version=2)#pts,all=loadPoints("./data/mensuel/coords/")
#     #plotPtsLoc(pts)
#     #plotData(info["rc"][2],all)
#     #--
#     lats_, lons_, xx, yy = getMesh(all)
#     #print("lats_ =>", lats_,"\nlons =>", lons_)
#     #test = reMesh(analyse["l2"][info["rc"][k-2],:],all["lats"],all["lons"],lats_,lons_)
#     #print("lats =>", all["lats"],"\nlons =>", all["lons"])
#     #print("lats_ =>", lats_,"\nlons =>", lons_)
#     #"""

    
#     cpt=k-2
#     #for cpt in range(k):
#     getIntraAnnual(kmed["ed"][:,cpt],mth="KMED-ED",path=url+"res/",day=False, lim=.1)
#     #getIntraAnnual(kmed["l2"][:,cpt],mth="KMED-L2",path=url+"res/")
#     #getIntraAnnual(ahc["ed"][:,cpt],mth="AHC-ED",path=url+"res/",day=False, lim=.3)
#     #getIntraAnnual(ahc["l2"][:,cpt],mth="AHC-L2",path=url+"res/")
#     plt.close('all')
        
#     #for cpt in range(k):
#     getInterAnnual(kmed["ed"][:,cpt],mth="KMED-ED",path=url+"res/",day=False, lim=.2)
#     #getInterAnnual(kmed["l2"][:,cpt],mth="KMED-L2",path=url+"res/")
#     #getInterAnnual(ahc["ed"][:,cpt],mth="AHC-ED",path=url+"res/",day=False, lim=.1)
#     #getInterAnnual(ahc["l2"][:,cpt],mth="AHC-L2",path=url+"res/")
#     plt.close('all')
#     #"""

#     """
#     for f in range(4):
#         test = reMesh(analyse["l2"][info["rc"][f]],all["lats"],all["lons"],lats_,lons_)
#         plt.figure()
#         plt.imshow(test,origin='lower')
#         plt.show()
#     """
#     #"""
#     for f in range(k):
#         m,a = getMonthById(info["rc"][f],1981)
#         d=f"{m:02}/{a}"
#         #d,_,_ = getDateById(info["rc"][f]+1,1981,2020)
#         n = info["n"][f]
#         test = reMesh(analyse["l2"][info["rc"][f],:],all["lats"],all["lons"],lats_,lons_)
#         plotData(test,xx,yy,title=f"KMED-ED-K{k}-C{f+1} ({d}) [{n}]",fname=f"rc_KMED-ED-K{k}-C{f+1}",path=url+"res/",cmax=400)

#     """info = getClusterInfo(analyse["ed"], ahc["ed"][:,k-2], ed=True)
#     for f in range(k):
#         #m,a = getMonthById(info["rc"][f],1981)
#         #d=f"{m:02}/{a}"
#         d,_,_ = getDateById(info["rc"][f]+1,1981,2020)
#         n = info["n"][f]
#         test = reMesh(analyse["l2"][info["rc"][f],:],all["lats"],all["lons"],lats_,lons_)
#         plotData(test,xx,yy,title=f"AHC-ED-K{k}-C{f+1} ({d}) [{n}]",fname=f"rc_AHC-ED-K{k}-C{f+1}",path=url+"res/", cmax=400)
#     #"""
#     plt.close('all')

#     #"""
#     info = getClusterInfo(analyse["l2"], kmed["ed"][:,k-2])
#     for f in range(k):
#         test = reMesh(info["cc"][f],all["lats"],all["lons"],lats_,lons_)
#         n = info["n"][f]
#         plotData(test,xx,yy,title=f"KMED-ED-K{k}-C{f+1} ({n})",fname=f"cc_KMED-ED-K{k}-C{f+1}",path=url+"res/",cmax=400)
    
#     """info = getClusterInfo(analyse["l2"], ahc["ed"][:,k-2])
#     for f in range(k):
#         test = reMesh(info["cc"][f],all["lats"],all["lons"],lats_,lons_)
#         n = info["n"][f]
#         plotData(test,xx,yy,title=f"AHC-ED-K{k}-C{f+1} ({n})",fname=f"cc_AHC-ED-K{k}-C{f+1}",path=url+"res/",cmax=400)
    
#     #"""
#     plt.close('all')

# def main2():

#     url = "./data/journalier/"
#     analyse = loadAnalyse(url, load=True, version=2)
#     #storeDistMatrix(analyse["l2"],path=url)
#     storeDistMatrix(analyse["ed"],path=url,ed=True)
#     exit(0)
#     kmed,ahc = clusterData(analyse, path=url, load=False)

#     #"""
#     #--
#     plt.figure(figsize=(8,6))
#     plt.rcParams.update({'font.size': 16})
#     getSilhouette(analyse["ed"], kmed["ed"], ed=True, path=url, mth="kmed-ed", style="r.-")
#     getSilhouette(analyse["ed"], ahc["ed"], ed=True, path=url, mth="ahc-ed", style="r.--")
#     getSilhouette(analyse["l2"], kmed["l2"], path=url, mth="kmed-l2", style="k.-")
#     getSilhouette(analyse["l2"], ahc["l2"], path=url, mth="ahc-l2", style="k.--", save=True, fname="all")
#     plt.close('all')
#     #---
#     plt.figure(figsize=(8,6))
#     plt.rcParams.update({'font.size': 16})
#     getCalHar(analyse["ed"], kmed["ed"], ed=True, path=url, mth="kmed-ed", style="b.-")
#     getCalHar(analyse["ed"], ahc["ed"], ed=True, path=url, mth="ahc-ed", style="b.--")
#     getCalHar(analyse["l2"], kmed["l2"], path=url, mth="kmed-l2", style="k.-")
#     getCalHar(analyse["l2"], ahc["l2"], path=url, mth="ahc-l2", style="k.--", save=True, fname="all")
#     plt.close('all')
#     #--
#     #"""

#     #"""
#     k=8
#     info = getClusterInfo(analyse["ed"], kmed["ed"][:,k-2], ed=True)
#     #print(calharED(analyse["ed"], kmed["ed"][:,2]))
#     pts,all=loadPoints(url,version=2)#pts,all=loadPoints("./data/mensuel/coords/")
#     #plotPtsLoc(pts)
#     #plotData(info["rc"][2],all)
#     #--
#     lats_, lons_, xx, yy = getMesh(all)
#     #print("lats_ =>", lats_,"\nlons =>", lons_)
#     #test = reMesh(analyse["l2"][info["rc"][k-2],:],all["lats"],all["lons"],lats_,lons_)
#     #print("lats =>", all["lats"],"\nlons =>", all["lons"])
#     #print("lats_ =>", lats_,"\nlons =>", lons_)
#     #"""


#     #"""
#     for cpt in range(k):
#         getIntraAnnual(kmed["ed"][:,cpt],mth="KMED-ED",path=url+"res/")
#         #getIntraAnnual(kmed["l2"][:,cpt],mth="KMED-L2",path=url+"res/")
#         getIntraAnnual(ahc["ed"][:,cpt],mth="AHC-ED",path=url+"res/")
#         #getIntraAnnual(ahc["l2"][:,cpt],mth="AHC-L2",path=url+"res/")
#         plt.close('all')
        
#     for cpt in range(k):
#         getInterAnnual(kmed["ed"][:,cpt],mth="KMED-ED",path=url+"res/")
#         #getInterAnnual(kmed["l2"][:,cpt],mth="KMED-L2",path=url+"res/")
#         getInterAnnual(ahc["ed"][:,cpt],mth="AHC-ED",path=url+"res/")
#         #getInterAnnual(ahc["l2"][:,cpt],mth="AHC-L2",path=url+"res/")
#         plt.close('all')
#     #"""

#     """
#     for f in range(4):
#         test = reMesh(analyse["l2"][info["rc"][f]],all["lats"],all["lons"],lats_,lons_)
#         plt.figure()
#         plt.imshow(test,origin='lower')
#         plt.show()
#     """
#     #"""
#     for f in range(k):
#         m,a = getMonthById(info["rc"][f],1981)
#         test = reMesh(analyse["l2"][info["rc"][f],:],all["lats"],all["lons"],lats_,lons_)
#         plotData(test,xx,yy,title=f"KMED-ED-K{k}-C{f+1} ({m:02}/{a})",fname=f"rc_KMED-ED-K{k}-C{f+1}",path=url+"res/")

#     info = getClusterInfo(analyse["ed"], ahc["ed"][:,k-2], ed=True)
#     for f in range(k):
#         m,a = getMonthById(info["rc"][f],1981)
#         test = reMesh(analyse["l2"][info["rc"][f],:],all["lats"],all["lons"],lats_,lons_)
#         plotData(test,xx,yy,title=f"AHC-ED-K{k}-C{f+1} ({m:02}/{a})",fname=f"rc_AHC-ED-K{k}-C{f+1}",path=url+"res/")
#     #"""
#     plt.close('all')

#     #"""
#     info = getClusterInfo(analyse["l2"], kmed["ed"][:,k-2])
#     for f in range(k):
#         test = reMesh(info["cc"][f],all["lats"],all["lons"],lats_,lons_)
#         n = info["n"][f]
#         plotData(test,xx,yy,title=f"KMED-ED-K{k}-C{f+1} ({n})",fname=f"cc_KMED-ED-K{k}-C{f+1}",path=url+"res/",cmax=500)
    
#     info = getClusterInfo(analyse["l2"], ahc["ed"][:,k-2])
#     for f in range(k):
#         test = reMesh(info["cc"][f],all["lats"],all["lons"],lats_,lons_)
#         n = info["n"][f]
#         plotData(test,xx,yy,title=f"AHC-ED-K{k}-C{f+1} ({n})",fname=f"cc_AHC-ED-K{k}-C{f+1}",path=url+"res/",cmax=500)
    
#     #"""
#     plt.close('all')
    
# #Run main
# #main()
