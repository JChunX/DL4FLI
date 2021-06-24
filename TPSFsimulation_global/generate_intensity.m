function intensity = generate_intensity(image,pc)
% generate random intensity for input binary image
% pc - photon count ranges: [lower upper]
    m = size(image, 1);
    n = size(image, 2);
%     random matrix of intensity values possessing values within maximum
%     photon count threshold.
    int1 = rand(m, n) * (pc(2)-pc(1)) + pc(1);
    intensity = int1.*image;
end