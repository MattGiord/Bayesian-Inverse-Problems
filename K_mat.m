function k=K_mat(x,y,v,length)
    if x==y
        k = 1;
    else
    dist=norm(x-y,2);
    part1=2^(1-v)/gamma(v);
    part2=(sqrt(2*v)*dist/length)^v;
    part3=besselk(v,sqrt(2*v)*dist/length);
    k = part1*part2*part3;
    end
end