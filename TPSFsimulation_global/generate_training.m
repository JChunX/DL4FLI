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
N_total = 1000;
nTG = 256;
photon_count = [250 1500];
tau1_range = [0.4,0.7];
tau2_range = [2.0,3.0];
pathN = 'D:\Data\DL-FLIM\test_global';

if ~exist(pathN, 'dir')
   mkdir(pathN)
end

k = 1;
while k <= N_total
% Take 28x28 subset of random 32x32 MNIST image
    im_binary = train_images(3:end-2,3:end-2,round(rand()*(size(train_images,3)-1))+1);
% Make sure it is not too sparse (we want voxels with more TPSFs than
% less)
    if sum(sum(im_binary)) < 250
        continue
    end
% Generate intensity image map
    inten = generate_intensity(im_binary, photon_count);
% Generate t1, t2 and AR image maps
    [tau1, tau2, ratio] = generate_lifetime(im_binary, tau1_range, tau2_range);
    [data, irf] = generate_tpsfs(inten, nTG, tau1, tau2, ratio, irf_whole);
    m = size(im_binary,1);
    n = size(im_binary,2);
    
    t1 = max(tau1,[],'all');
    t2 = max(tau2,[],'all');
    rT = ratio;
    sigD = data;
    I = inten.*im_binary;
        
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

    save(filenm, 'sigD', 'I', 't1', 't2', 'rT', 'irf', '-v7.3');
    k = k+1;
end
