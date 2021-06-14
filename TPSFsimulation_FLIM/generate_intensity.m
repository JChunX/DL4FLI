function intensity = generate_intensity( image )
% generate random intensity for input binary image
    m = size(image, 1);
    n = size(image, 2);
%     random matrix of intensity values possessing values within maximum
%     photon count threshold.
    int1 = rand(m, n) * 1500 + 500; % 500 - 2000 p.c.
    intensity = int1.*image;
end