function [full_data0, irf] = generate_tpsfs(intensity, nTG, tau1, tau2, ratio, irf_whole)
% dimension: image * time gate (28x28xnTG)
% irf: unit Instrumental Response Function (sum=1)

M = size(intensity, 1);
N = size(intensity, 2);
% Number of time-points/gates
width = 4.89e-2; % Different time-point durations for different apparatus settings
time = [1:1:nTG]*width;

% Pre-allocate memory for each TPSF voxel
full_data0 = zeros(M,N,nTG);

% Grab IRF from library
irf = irf_whole(:,round(rand()*(size(irf_whole,2)-1))+1);   

% Loop over all pixels spatially
parfor i=1:M
    for j=1:N
%         Only loop at locations from which TPSFs can be created.
        if tau1(i,j)~=0
%             Create initial bi-exponential given the tau1, tau2 and ratio
%             values at the image position (i,j)
            decay = ratio(i,j)*exp(-time./tau1(i,j))+(1-ratio(i,j))*exp(-time./tau2(i,j));  
%             Convolve IRF with our exp. decay
            decay = conv(decay,irf/sum(irf));
%             Sample back to the original number of time-points by including random
%             effects due to laser-jitter (point of TPSF ascent).
            r = rand();
            if r > .75
                decay = decay(1:nTG);
            elseif r < .25
                rC = round(rand().*3);
                decay = [zeros(rC,1); decay(1:nTG-rC)];
            else
                rC = round(rand().*3);
                decay = decay(1+rC:nTG+rC);
            end
%             Multiple the decay by its corresponding intensity value
%             (maximum photon count)
            decay = decay*intensity(i,j);
%             Add poisson noise
            cur = round(poissrnd(decay));
%             Assign the decay to its corresponding pixel location
            full_data0(i,j,:) = cur/max(cur);
        end
    end
end

end