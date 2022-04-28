import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.signal import find_peaks


from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv, spsolve
from scipy.interpolate import interp1d,LSQUnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


# -------------------------------------------------------------
#           RUN LSD
# -------------------------------------------------------------


def runLSD(values, row_no, col_no, n, m, weights, fluxes):
    # MATRIX CALCULATIONS USING 'csc_matrix' TO IMPROVE EFFICIENCY
    # The uncertainties are not computed here (necessitates inverting the MtWM matrix

    # See Lienhard et al. 2022 eq. 4 

    M = csc_matrix((values, (row_no, col_no)), shape=(n, m))
    Mt = M.transpose()
    MtW = Mt.dot(csc_matrix((weights, (np.arange(n), np.arange(n))), shape=(n, n)))
    MtWM = MtW.dot(M)

    A = MtWM

    B = MtW.dot(csc_matrix(fluxes).transpose())

    return spsolve(A, B, use_umfpack=True)


def runLSD_inv(values, row_no, col_no, n, m, weights, fluxes):
    # MATRIX CALCULATIONS USING 'csc_matrix' TO IMPROVE EFFICIENCY
    # The uncertainties are computed here (necessitates inverting the MtWM matrix

    M = csc_matrix((values, (row_no, col_no)), shape=(n, m))
    Mt = M.transpose()
    MtW = Mt.dot(csc_matrix((weights, (np.arange(n), np.arange(n))), shape=(n, n)))
    MtWM = MtW.dot(M)

    A = MtWM

    B = MtW.dot(csc_matrix(fluxes).transpose())
    #Z = spsolve(A, B, use_umfpack=True)
    inverse = inv(csc_matrix(MtWM))
    Z = inverse.dot(B)
    return np.asarray(Z.todense()).ravel(), np.sqrt(np.diag(inverse.todense()))



# -------------------------------------------------------------
#           OTHER FUNCTION USED LATER
# -------------------------------------------------------------



def overlaps(lams, lamlist, maxdist):
    # lams = wvls in order (1d)
    # lamlist = wvls of abs lines in line list (i.e. VALD)

    # (lam1, lam2 ... lamn)
    # (lam1, lam2 ... lamn)
    lams_tile = np.tile(lams, (len(lamlist), 1))

    # (lamlist1, lamlist1 ... lamlist1)
    # (lamlist2, lamlist2 ... lamlist2)
    lamlist_tile = np.tile(lamlist, (len(lams), 1)).transpose()

    dist_to_lamlistval = np.abs(lams_tile - lamlist_tile)

    # lam_i overlaps with nr_of_overlaps[i] absorption lines in lamlist by less/equal maxdist
    if np.size(maxdist) == 1:
        nr_of_overlaps = np.sum((dist_to_lamlistval < maxdist) * 1.0, axis=0)
    else:
        maxdist_tile = np.tile(maxdist, (len(lams), 1)).transpose()
        nr_of_overlaps = np.sum((dist_to_lamlistval < maxdist_tile) * 1.0, axis=0)

    return nr_of_overlaps


def Gaussian(x, A, x0, sigma, B):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + B


def upper_envelope(x, y):
    #used to compute the tapas continuum. find peaks then fit spline to it.
    peaks = find_peaks(y, height=0.2, distance=len(x) // 500)[0]
    # t= knot positions
    spl = LSQUnivariateSpline(x=x[peaks], y=y[peaks], t=x[peaks][5::10])
    return spl(x)



# -------------------------------------------------------------
#           CLASSES
# -------------------------------------------------------------



class stellar_parameters:
    #save stellar parameters from vald3 here
    def __init__(self, star,valddir,dirdir,pipname,c):
        self.star = star
        self.valddir = valddir
        self.dirdir = dirdir
        self.pipname = pipname        
        self.c = c
        
    def loadVALD(self,VALDfile):
        #load data from standard VALD3 output from “Extract Stellar” in the “Long format”.

        counter = 0
        for line in reversed(open(VALDfile).readlines()):
            if len(line.split(",")) > 10:
                break
            counter += 1
        num_lines = sum(1 for line in open(VALDfile))
        last_line = num_lines - counter + 3

        with open(VALDfile) as f_in:
            x = np.genfromtxt(itertools.islice(f_in, 3, last_line, 4), dtype=str,delimiter=',')

        self.element = x[:,0]
        self.VALDlambdas = x[:,1].astype(float)
        self.VALDdepths = x[:,13].astype(float)

        try:
            #these are not needed for the analysis
            self.loggf = x[:,2].astype(float)
            self.elow = x[:,3].astype(float)
            self.eup = x[:,5].astype(float)
            self.rad = x[:,10].astype(float)
            self.stark = x[:,11].astype(float)
            self.waals = x[:,12].astype(float)
            self.mean_lande = x[:,9].astype(float)
            self.mean_lande[self.mean_lande==99] = 0
        except:
            pass

    def VALD_data(self):
        try:
            #find valdfile for this star
            valdfile = self.valddir + f"{self.star}.txt" 
            self.loadVALD(valdfile)
        except:
            valdfile = self.valddir + "Sun.txt"
            self.loadVALD(valdfile)
        print("loaded", valdfile)
        #set lande factor to 0 if not in database

  
        #if off: wavelengths must be "in air". check VALD request.
        #return VALDlambdas, VALDdepths, mean_lande,loggf, rad, stark, waals,elow,eup,element

    def inspect_data(self,ii,spectrum,wavelengths,start,window):

        #-----------------------------------------------------
        # read a spectrum and its corresponding data
        #info = pd.read_csv(dirdir+"Info"+str(ii)+".csv")
        info = pd.read_csv(self.dirdir+"Info.csv")

        rv_ccf = info["rv_ccf"][0]
        bjd = info["bjd_ccf"][0]

        #-----------------------------------------------------
        # plot absorption lines in VALD and spectrum in stellar rest frame

        # choose a wvl window
        end = start + window

        conversion2 = 1.+rv_ccf/self.c

        plt.figure(figsize=(8,4))
        plt.plot(wavelengths.flatten()/conversion2,spectrum.flatten(),".",label="RASSINE-corrected spectrum")
        plt.plot(self.VALDlambdas,1.-self.VALDdepths,"D",markersize=5,label="VALD absorption line")

        plt.ylim(-1+1,0.5+1)
        plt.xlim(start,end)
        plt.xlabel("Wavelength [\AA]",fontsize=15)
        plt.ylabel("Relative Intensity",fontsize=15)
        plt.legend(fontsize=15)

class analyse:
    def __init__(self,c,VALDlambdas,VALDdepths):      
        self.c = c
        self.VALDlambdas = VALDlambdas 
        self.VALDdepths = VALDdepths

    def prep_spec(self,ii, datafile, erroption):
        """
        Get spectrum, associated wavelengths, and uncertainties, tellurics. Compute weights. ONLY TO GET SOME FIRST INFORMATION ABOUT THE SPECTRUM RV AND COMMON PROFILE.
        Parameters
        ----------
        ii : int
            identifier of spectrum
        datafile : dict
            contains all the information for the spectrum

        Output
        ----------
        spectrum : array orders x pixels
            Spectrum as input except that nans are set to 0 (and 0 if -1 in telluric model (for EXPRES))
        wavelengths: array orders x pixels
            wavelengths from datafile
        weights: array orders x pixels
            Weights for LSD (1/err**2, 0 if affected by telluric (according to criterion), 0 if flux above continuum by 5 per cent or 0.1 below -1 (i.e. negative flux)
        """
        # version of prep_spec3 (below) with less options and no q_map. just define wvl, spec, and weight matrices. (excluding deep tellurics, nans, and outliers)
        spectrum = np.copy(datafile["spectrum"][ii])
        tps_tellurics = self.tapas_tellurics[ii]

        spectrum = (spectrum + 1.0) / tps_tellurics - 1.0

        wavelengths = np.copy(datafile["wavelengths"][ii])
        if erroption == 0:
            weights = 1.0 / datafile["err"][ii] ** 2
        if erroption == 1:
            weights = 1.0 / datafile["err_envelope"][ii] ** 2


        #CHANGED
        weights[tps_tellurics < 0.9] = 0
        spectrum[tps_tellurics == -1] = 0

        weights[np.isnan(spectrum)] = 0.0
        spectrum[np.isnan(spectrum)] = 0.0

        weights[spectrum > self.exclupper] = 0
        weights[spectrum < self.excllower] = 0

        self.weights = weights
        self.spectrum = spectrum
        self.wavelengths = wavelengths
        self.nr_of_orders,self.nr_of_pixels = np.shape(spectrum)


    def worker(self,order,vel):
        """
        For given order: perform LSD and extract the common profile, common profile uncertainties. Compute convolution model 
        
        Parameters
        ----------
        order : int
            First array: periods [d]
        spectrum : array orders x pixels
            Fluxes
        wavelengths : array orders x pixels
            Wavelength corresponding to the fluxes
        weights : array orders x pixels
            Weights of the individual fluxes
        self.VALDlambda : array
            Central wavelength of absorption lines (VALD3)
        self.VALDdepths : array
            Depth of absorption lines (VALD3)
        vel : array
            Velocity grid to run LSD on (velocity grid for common profile)
            
        Output:
        Z : array
            common profile
        M.dot(Z) : array
            convolution model
        """ 
        #LSD for a given order
        spectrum_o = self.spectrum[order,:]
        wavelengths_o = self.wavelengths[order,:]
        weights_o = self.weights[order,:]         
        #value, row, column, reject = convolmat9.convolmat9(wavelengths_o, vel, self.VALDlambda,self.VALDdepths,np.zeros((len(self.VALDlambda))), 0,0,0,1.,vel[1]-vel[0])
        value, row, column = self.cvmt(wavelengths_o, vel, self.VALDlambdas,self.VALDdepths)
            #print(len(value),len(row),len(column))
        M = csc_matrix((value,(row,column)), shape=(len(wavelengths_o), len(vel)))
        
        Z = runLSD(value, row, column, len(wavelengths_o), len(vel), weights_o, spectrum_o)

        return Z,M.dot(Z)


    def get_wide_lines(self):
        #exclude wide lines too (such as h alpha?)
        #find wide lines. different options.
        #set to 0, i.e. no wide lines are excluded! (-> this cell can be ignored)

        self.wide_lines = []   

        #exclude wide lines.
        if self.exclwidelinesparam==1:
            #exclude very wide line (eg. H-alpha)
            self.wide_lines = np.copy(self.VALDlambdas)[(stark>-2) & (stark!=0)]
            #the following abs line regions are problematic (wide lines by eye)
            self.wide_lines = np.hstack((wide_lines,np.array([5172.5,5183.7,5890,5896])))

        #convert to stellar reference frame to match spectral absorption lines
        #wide lines: in stellar rest frame
        #wvls: in barycentric rest frame

        if len(self.wide_lines)>0:
            self.wide_lines /= self.barycentric_to_stellar_restframe[iis[test_ii]]
        


    def get_q_map(self,info_file):
        #get the quality map based on the input above.
        q_map = np.zeros((np.shape(self.spectrum)))

        for order in range(np.shape(self.spectrum)[0]):
            wvl_h_order = self.wavelengths[order, :]
            div_order = self.div[order, :]


            peaks = find_peaks(div_order, height=self.modelspecdeviationcut)[0]


            ovlp = overlaps(
                wvl_h_order,
                wvl_h_order[peaks],
                self.alldata["absline_halfwidth_include"] * np.mean(wvl_h_order),
            )


            # check if the current order has any wide_lines in it
            if ((self.wide_lines > wvl_h_order.min()) & (self.wide_lines < wvl_h_order.max())).any():
                lines_to_remove = self.wide_lines[
                    ((self.wide_lines > wvl_h_order.min()) & (self.wide_lines < wvl_h_order.max()))
                ]

                # remove plus/minus 3.5 angstrom (empirical)
                ovlp_rem = overlaps(wvl_h_order, lines_to_remove, 3.5)

                # exclude all measurement where the model does not agree with the data or where the measurements are impacted by wide lines
                exclude = np.where((ovlp != 0) | (ovlp_rem != 0))[0]
                rem_line = True
            else:
                exclude = np.where(ovlp != 0)[0]
                rem_line = False

            # define quality map. bad = 1
            q_map[order, exclude] = 1

        #also exclude wavelengths that are only sometimes visible due to barycentric motion (i.e. borders)

        bervmin = np.where(info_file["berv"]==info_file["berv"].min())[0][0]
        bervmax = np.where(info_file["berv"]==info_file["berv"].max())[0][0]

        for order in np.arange(self.nr_of_orders):
            wvlmin = self.alldata["wavelengths"][bervmax][order][0]
            wvlmax = self.alldata["wavelengths"][bervmin][order][-1]
            wo = np.where((self.wavelengths[order,:]>wvlmax) | (self.wavelengths[order,:]<wvlmin))[0]
            q_map[order,:][wo] = 1


        #only include fluxes where there is an accepted line and no line deeper than maxdepthparam
        for order in np.arange(self.nr_of_orders):
            wavelengths_o = self.wavelengths[order,:]

            inthisorder = np.where((self.VALDlambdas>wavelengths_o.min()-3) & (self.VALDlambdas<wavelengths_o.max()+3))[0]
            VALDlambdas_here = self.VALDlambdas[inthisorder]
            VALDdepths_here = self.VALDdepths[inthisorder]

            #choose region (convert to stellar rest frame to match with VALD)
            #only regions with abslines with depth at least mindepth
            overlap = overlaps(wavelengths_o,VALDlambdas_here[VALDdepths_here>self.mindepthparam]/self.barycentric_to_stellar_restframe[self.test_ii],self.alldata["absline_halfwidth_include"]*np.mean(wavelengths_o))



            overlap2 = overlaps(wavelengths_o,VALDlambdas_here[VALDdepths_here>self.maxdepthparam]/self.barycentric_to_stellar_restframe[self.test_ii],self.alldata["absline_halfwidth_include"]*np.mean(wavelengths_o))
            
            nonselection = np.where(~((overlap2==0) & (overlap>=1)))[0]
            
            q_map[order,nonselection]=1
            self.q_map = q_map

            #add quality map to alldata dictionary 
            self.alldata["q_map"]={}

            for ii in self.iis:
                self.alldata["q_map"][ii]=np.zeros((np.shape(q_map)))

            for order in np.arange(self.nr_of_orders):

                wvl_h_order = self.wavelengths[order,:]

                #interpolate to wvls of other spectra
                fct = interp1d(wvl_h_order,self.q_map[order,:],bounds_error=False,fill_value = 1)
                for ii in self.iis:
                    wvlo = self.alldata["wavelengths"][ii][order]
                    self.alldata["q_map"][ii][order,:] = np.round(fct(wvlo))
                    #uncertainty = 0? suspicious.
                    zero_error = np.where(self.alldata["err"][ii][order,:]==0)[0]
                    self.alldata["q_map"][ii][order,:][zero_error]=1



    def get_t_map(self):
        self.alldata["t_map"]={}

        barytelluricmap = np.zeros(np.shape(self.spectrum))
        barytelluric_referencewvl = self.alldata["wavelengths"][self.test_ii]

        ########

        #exclude any barycentric wavelength if it is ever affected by telluric line deeper than telluric_cut

        ########

        for order in np.arange(self.nr_of_orders):
            #reinterpolate to pixel space of first spectrum. -> same wvls in barycentric frame will be excluded for all spectra
            
            for ii in self.iis:
                #bary wavelengths affected by tellurics for this spectrum and order
                x= self.alldata["wavelengths"][ii][order]
                y = np.zeros((self.nr_of_pixels))
                y[self.tapas_tellurics[ii][order]<(1.-self.telluric_cut)]=1
                
                fct = interp1d(x,y,bounds_error=False,fill_value = 1)
                #interpolate to barytelluric_referencewvl

                #deep_tellurics_in_pixel_space_of_reference_spectrum
                dtipsors = np.round(fct(barytelluric_referencewvl[order,:]))
                barytelluricmap[order,:][dtipsors==1] = 1

        #initialise   
        for ii in self.iis:
            self.alldata["t_map"][ii] = np.zeros(np.shape(self.spectrum))

            
        #interpolate to pixels

        for order in np.arange(self.nr_of_orders):
            #reinterpolate to barycentric wvls of other spectra. -> same wvls in barycentric frame will be excluded for all spectra
            fct = interp1d(barytelluric_referencewvl[order,:],barytelluricmap[order,:],bounds_error=False,fill_value = 1)
            for ii in self.iis:
                wvlo = self.alldata["wavelengths"][ii][order]
                if self.telloption==1:
                    self.alldata["t_map"][ii][order,:] = np.round(fct(wvlo))
                else:
                    self.alldata["t_map"][ii][order,:][self.tapas_tellurics[ii][order]<(1.-self.telluric_cut)]=1


    def show_map(self):



        plt.figure(figsize=(10,10))
        plt.xlabel("Pixel number",fontsize=20)
        plt.ylabel("Spectral order",fontsize=20)

        
        
        qm = self.alldata["q_map"][self.test_ii]


        for order in range(self.nr_of_orders):
            #plt.plot(np.arange(len(qm[order,:]))[qm[order,:]>0],order*qm[order,:][qm[order,:]>0],".",color="blue")
            if order == 0:
                plt.plot(np.arange(len(qm[order,:]))[qm[order,:]>0],order*np.ones((self.nr_of_pixels))[qm[order,:]>0],".",markersize=4,color="cornflowerblue",label="No line / model-spectrum deviation")
            else:
                plt.plot(np.arange(len(qm[order,:]))[qm[order,:]>0],order*np.ones((self.nr_of_pixels))[qm[order,:]>0],".",markersize=4,color="cornflowerblue")
        


        qm = self.alldata["t_map"][self.test_ii]
        for order in range(self.nr_of_orders):
            if order == 0:
                plt.plot(np.arange(len(qm[order,:]))[qm[order,:]>0],order*qm[order,:][qm[order,:]>0],".",markersize=4,color="palevioletred",label="Telluric / Edge pixel")
            else:
                plt.plot(np.arange(len(qm[order,:]))[qm[order,:]>0],order*qm[order,:][qm[order,:]>0],".",markersize=4,color="palevioletred")
        
        
        qm = self.alldata["q_map"][self.test_ii]
        for order in range(self.nr_of_orders):
            if order == 0:
                plt.plot(np.arange(len(qm[order,:]))[qm[order,:]==0],order*np.ones((self.nr_of_pixels))[qm[order,:]==0],".",markersize=4,color="navajowhite",label="Included")
            else:
                plt.plot(np.arange(len(qm[order,:]))[qm[order,:]==0],order*np.ones((self.nr_of_pixels))[qm[order,:]==0],".",markersize=4,color="navajowhite")

        plt.legend()
        plt.savefig(self.resdir+f"q_map.png")













    def get_tapas_transmittance(self, pipname, transmittance_csv, info_file):

        t2_exists = False


        for ii in self.iis:
            if pipname == "EXPRES":
                self.tapas_tellurics[ii] = np.load(
                    self.dirdir + "telluric_" + str(ii) + ".npy"
                )
            else:
                no_tapas_file = False

                # -----------------------------------------------------
                # get tapas file for spectrum with identifier ii
                if transmittance_csv:
                    transmittance_data = pd.read_csv(transmittance_csv)
                    # assume that transmittance given in nm
                    tapas_wvl = transmittance_data["x"] * 10
                    tapas_trans = transmittance_data["y"]
                    f = interp1d(tapas_wvl, tapas_trans, bounds_error=False)
                else:
                    try:
                        info = pd.read_csv(self.dirdir + "Info" + str(ii) + ".csv")
                        date_here = info["dateobs"][0].replace("T", " ")

                        tapas_filedate = (
                            date_here.replace("-", "")
                            .replace(":", "")
                            .replace(" ", "_")[:13]
                        )
                        tapas = fits.open(
                            f"./tellurics_fits/{self.star}/tapas_" + tapas_filedate + ".fits"
                        )
                    except Exception as e:
                        # print("No tapas file")
                        tapas = fits.open("./tellurics_fits/star/tapas_000001.fits")
                        no_tapas_file = True

                    tapas_wvl = (tapas[1].data["wavelength"]) * 10.0
                    tapas_trans = tapas[1].data["transmittance"]
                    tapas.close()
                    tapas_wvl = tapas_wvl[::-1]
                    tapas_trans = tapas_trans[::-1]

                    assert (sorted(tapas_wvl) == tapas_wvl).any(), "not sorted"
                    # -----------------------------------------------------
                    # continuum normalise tapas transmittance

                    if no_tapas_file and t2_exists:
                        background = np.copy(background_t2)
                    else:
                        background = upper_envelope(tapas_wvl, tapas_trans)

                    f = interp1d(
                        tapas_wvl, tapas_trans / background, bounds_error=False
                    )
                # -----------------------------------------------------
                # interpolate tapas transmittance for spectral wavelengths as in observatory rest frame

                # conversion to go from barycentric to observatory frame
                conversion = 1.0 + info_file["berv"][ii] / self.c

                # interpolate tapas for wvls in telluric frame
                tell_reinterp = f((self.alldata["wavelengths"][ii]) / conversion)

                self.tapas_tellurics[ii] = tell_reinterp
                # -----------------------------------------------------

                if not t2_exists and no_tapas_file:
                    # save this background for the next spectrum without its own tapas file
                    background_t2 = np.copy(background)
                    # now there is a background computed for the t2 tapas spectrum. reuse.
                    t2_exists = True
        return 0






















    def cvmt(self,wavelengths_o, vel, vlambda,vdepth):
        dv = vel[1]-vel[0]

        lenk = 3
        
        row2 = np.array([])
        column2 = np.array([])
        value2 = np.array([])
        
        b = len(wavelengths_o)//lenk
        
        for k in np.arange(lenk):
            #wvls in this chunk
            wavelengths_sub = wavelengths_o[b*k:b*(k+1)]

            #min wvl and max wvl to include
            lmin = wavelengths_sub.min()*(1.-(vel.max()+dv)/self.c)
            lmax = wavelengths_sub.max()*(1.-(vel.min()-dv)/self.c)
            
            #only include vald abslines that are withinn lmin and lmax
            vlambdaincl = vlambda[(vlambda>lmin) & (vlambda<lmax)]
            vdepthincl = vdepth[(vlambda>lmin) & (vlambda<lmax)]
            
            #compute velocity shift between vald lines and wvls
            v_shift = self.c*(np.tile(wavelengths_sub,(len(vlambdaincl),1)) / (np.tile(vlambdaincl,(len(wavelengths_sub),1))).transpose()-1.)

            #only keep those with velocity shift within velocity grid
            # l = index of vald absorption line
            # i = index of affected flux/wvl
            l,i = np.where((v_shift<=(vel.max()+dv)) & (v_shift>=(vel.min()-dv)))
            value0 = v_shift[l,i]
            
            #index shift b*k due to wvl-chunk-splitting for this look
            i += b*k

            #velocity position between two points of velocity grid
            pos = ((value0-vel[0])%dv)/dv
            #left velocity grid point
            jleft = (value0-vel[0])//dv
            
            #check which ones falll within velocity grid
            woleft = (jleft>=0) & (jleft<len(vel))
            woright = (jleft>=-1) & (jleft<len(vel)-1)
            
            #stack information
            i1 = np.hstack((i[woleft],i[woright]))
            j1 = np.hstack((jleft[woleft],jleft[woright]+1))

            #contribution to flux from each vald absorption line
            vstl1 = ((1.-np.hstack((pos[woleft],1.-pos[woright])))*vdepthincl[np.hstack((l[woleft],l[woright]))])
            
            sorter = i1+j1/len(vel)/10.
            indexorder = np.argsort(sorter)
            i1 = i1[indexorder]
            j1 = j1[indexorder]
            vstl1 = vstl1[indexorder]
            sorter = sorter[indexorder]
            uniquejs,whereuniquejindex,samejs = np.unique(sorter,return_inverse=True,return_index=True)

            row2 = np.hstack((row2,i1[whereuniquejindex]))
            value2 = np.hstack((value2,np.bincount(samejs,vstl1)))
            column2 = np.hstack((column2,j1[whereuniquejindex]))
        return value2,row2,column2








# -------------------------------------------------------------
#           GET APPROPRIATE WEIGHTS FOR LSD RUN
# -------------------------------------------------------------







def prep_spec3(datafile, ii,tapas_tellurics, erroption, usetapas):
    """
    Get spectrum, associated wavelengths, and uncertainties, tellurics. Compute weights. ONLY TO GET SOME FIRST INFORMATION ABOUT THE SPECTRUM RV AND COMMON PROFILE.
    Parameters
    ----------
    datafile : dict
        Contains all the information for the spectrum
    ii : int
        Identifier of spectrum
    weighting: str
        Which weighting scheme to use
    usetapas: Bool
        Divide by telluric model and mask deep tellurics?
    Output
    ----------
    weights: array orders x pixels
    spectrum : array orders x pixels
    wavelengths: array orders x pixels
        wavelengths from datafile
    """
    # lSD velocity grid
    vel = datafile["vel"]

    # data for later
    spectrum = np.copy(datafile["spectrum"][ii])
    wavelengths = np.copy(datafile["wavelengths"][ii])
    # -----------------------------------------------------

    # -----------------------------------------------------

    # choose weighting scheme


    if erroption == 0:
        weights = 1.0 / datafile["err"][ii] ** 2

    if erroption == 1:
        weights = 1.0 / datafile["err_envelope"][ii] ** 2

    if erroption == 2:
        err = np.transpose(np.tile(np.median(alldata["err"][ii],axis=1),(np.shape(alldata["err"][0])[1],1)))
        weights = 1.0 / err ** 2



    # -----------------------------------------------------

    # divide by telluric transmission spectrum

    tps_tellurics = tapas_tellurics[ii]

    if usetapas:
        spectrum = (spectrum + 1.0) / tps_tellurics - 1.0

    # set weight of data impacted by tellurics to 0

    weights[datafile["t_map"][ii] == 1] = 0
    # set spectrum to 0 here (divided by tellurics before). source of potential errors. better set to 0.
    spectrum[tps_tellurics == -1] = 0

    # set weights to 0 where data impacted by wide lines or model and data do not agree well (additivity etc.)

    weights[datafile["q_map"][ii] == 1] = 0
    # -----------------------------------------------------
    # exclude nans
    weights[np.isnan(spectrum)] = 0.0
    spectrum[np.isnan(weights)] = 0.0
    weights[np.isnan(weights)] = 0.0
    spectrum[np.isnan(spectrum)] = 0.0

    spectrum[np.isinf(weights)] = 0.0
    weights[np.isinf(weights)] = 0.0

    weights[datafile["err"][ii] < 0] = 0
    # exclude upper outliers
    weights[spectrum > 0.05] = 0
    weights[spectrum <= -1.0] = 0

    # -----------------------------------------------------
    return weights, spectrum, wavelengths



# -------------------------------------------------------------
#           EXTRACT RVS FROM COMMON PROFILES
# -------------------------------------------------------------



# analyse common spectra as saved in LSD_results. get RV.
def extract_rv_from_common_profiles(
    LSD_results,
    alldata,
    epoch_list,
    testorders,
    weight_orders,
    use_uncertainties,
):

    rv_all = []
    vel = alldata["vel"]
    Zs = []
    Zerrs = []

    for ii in epoch_list:



        # ----------------------------------------------------------
        # testing weighting (and normalising) the common profiles of the different orders before addition

        lsdres = np.copy(LSD_results[ii]["common_profile"])
        lsdreserr = np.copy(LSD_results[ii]["common_profile_err"])
        weightsum = 0
        common_profile_err = np.zeros((len(lsdres[0])))

        if weight_orders == "flux weight_fixed_throughout_time_series":
            
            pre_weights = 1.0 / (alldata["err_smoothed"][alldata["iis"][0]] ** 2)
            pre_weights[LSD_results[alldata["iis"][0]]["incl_map"] == 0] = 0

            for count, order in enumerate(testorders):

                order_weight = np.nanmean(pre_weights[order, :])

                weightsum += order_weight

                common_profile_err += order_weight ** 2 * lsdreserr[order, :] ** 2
                lsdres[order, :] = lsdres[order, :] * order_weight

            common_profile_err = common_profile_err / weightsum ** 2
            common_profile_err = np.sqrt(common_profile_err)


        if weight_orders == "flux weight_can_vary":
            for count, order in enumerate(testorders):

                order_weight = np.copy(alldata["order_weight"][order])

                weightsum += order_weight

                common_profile_err += order_weight ** 2 * lsdreserr[order, :] ** 2
                lsdres[order, :] = lsdres[order, :] * order_weight

            common_profile_err = common_profile_err / weightsum ** 2
            common_profile_err = np.sqrt(common_profile_err)

        Z = np.nansum(lsdres, axis=0) / weightsum


        Zs.extend([Z])
        Zerrs.extend([common_profile_err])

        # ----------------------------------------------------------
        # get first estimates of mean(/summed) common profile minimum

        # savitzky golay filter to smooth data

        yhat = savgol_filter(Z, max(int(int(len(vel) / 10) * 2 + 1), 5), 3)
        mean = vel[np.argmin(yhat)]

        # ----------------------------------------------------------
        # FIT GAUSSIAN TO THE DATA
        n = len(Z)
        wo = np.where(
            (vel > mean - alldata["sigmafit"]) & (vel < mean + alldata["sigmafit"])
        )[0]

        sigma_guess = alldata["initial_v_halfwidth"] / np.sqrt(np.log(2.0) * 2.0)

        try:
            if alldata["fitfunction"] == "Gaussian":
                if use_uncertainties:
                    popt, pcov = curve_fit(
                        Gaussian,
                        vel[wo],
                        Z[wo],
                        [-1, mean, sigma_guess, 0],
                        sigma=common_profile_err[wo],
                    )
                else:
                    popt, pcov = curve_fit(
                        Gaussian, vel[wo], Z[wo], [-1, mean, sigma_guess, 0]
                    )
            
        except Exception as e:
            print(e)
            popt = [-9999999, -9999999, -9999999, -9999999]

        # ----------------------------------------------------------
        # save results of different epochs to list

        rv_all.extend([popt[1]])



    rv_all = np.asarray(rv_all)
    rv_all *= 1000
    return rv_all, Zs, Z, Zerrs




def numerical_gradient(vel, profile):
    """
    Return the gradient of the profile.
    Parameters
    ----------
    vel : array
        The velocity values where the profile is defined.
    profile : array
        The values of the profile.
    Notes
    -----
    The gradient is computed using the np.gradient routine, which uses second
    order accurate central differences in the interior points and either first
    or second order accurate one-sides (forward or backwards) differences at
    the boundaries. The gradient has the same shape as the input array.
    """
    return np.gradient(profile, vel)


def RVerror(vel, profile, eprofile):
    """
    Calculate the uncertainty on the radial velocity, following the same steps
    as the ESPRESSO DRS pipeline.
    Parameters
    ----------
    vel : array
        The velocity values where the profile is defined.
    profile : array
        The values of the profile.
    eprofile : array
        The errors on each value of the profile.
    """
    profile_slope = numerical_gradient(vel, profile)
    profile_sum = np.sum((profile_slope / eprofile) ** 2)
    return 1.0 / np.sqrt(profile_sum)






