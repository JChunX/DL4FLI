function [decay] = generate_tpsfs(intensity, nTG, tau1, tau2, ratio, irf)


% Number of time-points/gates
width = 4.89e-2; % Different time-point durations for different apparatus settings
time = [1:1:nTG]*width;

decay = ratio*exp(-time./tau1)+(1-ratio)*exp(-time./tau2);  
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

decay = decay*intensity;
decay = round(poissrnd(decay));
decay = decay/max(decay);
end