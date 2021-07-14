% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Generates the simulated TPSF voxel data (FLIM) using functions included along with 
% IRF (deconvolved via software) and MNIST data.
% 
% Jason T. Smith, Rensselaer Polytechnic Institute, August 23, 2019
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% Workflow:
% 1. Generate FLIM voxels for train & test
% 2. upload train & test zips to drive
% 3. Run notebook

load FLIM_IRF;
load train_binary;

% Number of TPSF voxels to create
N_total = 200000;
nTG = 256;
low_photon_range = [50 250];
high_photon_range = [150000 250000];
tau1_range = [0.1,1.0];
tau2_range = [2.0,3.0];
pathN = 'C:\Users\xieji\Dropbox\Documents\Data\DL-FLIM\train_global_gan';

if ~exist(pathN, 'dir')
   mkdir(pathN)
end

k = 1;
while k <= N_total

% Generate t1, t2 and AR image maps
    I_low = rand*(low_photon_range(2)-low_photon_range(1)) + low_photon_range(1);
    I_high = rand*(high_photon_range(2)-high_photon_range(1)) + high_photon_range(1);
    [tau1, tau2, ratio] = generate_lifetime(tau1_range, tau2_range);
    irf = irf_whole(:,round(rand()*(size(irf_whole,2)-1))+1);   
    dk_low = generate_tpsfs(I_low, nTG, tau1, tau2, ratio, irf);
    dk_high = generate_tpsfs(I_high, nTG, tau1, tau2, ratio, irf);
    t1 = tau1;
    t2 = tau2;
    rT = ratio;
        
% Making sure sample numbers are assigned like 00001, 00002,.... 01001,
% 01002, etc.
    if k >=0 && k < 10
        n = ['0000' num2str(k)];
    elseif k >=10 && k<100
        n = ['000' num2str(k)];
    elseif k >=100 && k<1000
        n = ['00' num2str(k)];
    elseif k >=1000 && k<10000
        n = ['0' num2str(k)];
    else
        n = num2str(k);
    end
        
% Assign path along with file name.
    filenm = [pathN '\' 'a_' n '_' num2str(1)];

% Save .mat file. It is important to note the end '-v7.3' - this is one
% of the more convenient ways to facillitate easy python upload of 
% matlab-created data.

    save(filenm, 'dk_low', 'dk_high', 't1', 't2', 'rT', 'irf', '-v7.3');
    k = k+1;
end
