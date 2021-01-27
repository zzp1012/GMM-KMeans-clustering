using Distributions
using Random
using LinearAlgebra
using CSV

mutable struct myPoint
    x::Float64
    y::Float64
    close::Int16
    far::Int16
    row_num::Int16
    col_num::Int16
    key::Int64
end

mutable struct myGrid
    points::Array{myPoint}
end

mutable struct Centroid
    x::Float64
    y::Float64
    close_num::Int16
end

function assignPoint(i::Int, x::Float64, y::Float64, close::Int, far::Int, grid_size::Float64)
    row = floor(Int8, x/grid_size)
    col = floor(Int8, y/grid_size)
    max_num =  ceil(107 / grid_size)
    key = Int(row*max_num + col)
    myPoint(x, y, close, far, row, col, key)
end

function myGrid()
    myGrid([])
end

groupby(f, list::Array ) = begin
    foldl(list; init = Dict()) do dict, v
        push!(get!(dict, f(v), []), v)
        dict
    end
end

function get_point_key(p::myPoint)
    p.key
end

function get_centroid(points)
    x_sum = 0
    y_sum = 0
    count = 0
    close_num = 0
    for p in points
        x_sum += p.x *p.close
        y_sum += p.y *p.close
        close_num += p.close
        count += 1
    end
    Centroid(x_sum/close_num, y_sum/close_num, floor(close_num/count))
end

function sample_grid(one_centroid::Centroid, grid_size::Float64)
    r_tmp = rand(Uniform(0, grid_size/sqrt(pi)), one_centroid.close_num)
    theta_tmp = rand(Uniform(0, 2pi), one_centroid.close_num)
    X_tmp = one_centroid.x .+ cos.(theta_tmp) .* r_tmp
    Y_tmp = one_centroid.y .+ sin.(theta_tmp) .* r_tmp
    [X_tmp Y_tmp]
end

function generate_samples(project_path, filename)
    grid_size = 1.25
    arr = CSV.File(project_path * "/" * filename)
    grid = myGrid()
    for (i, row) in enumerate(arr)
        if row.Close == 0
            continue
        end
        point = assignPoint(i, row.X, row.Y, row.Close, row.Far, grid_size)
        push!(grid.points, point)
    end

    dict_1 = groupby(get_point_key, grid.points)
    centroids = map(get_centroid, collect(values(dict_1)))
    array_list = map(sample_grid, centroids, ones(length(centroids)) .+ grid_size)
    mapconcat2(v) = vcat((v)...)
    samples = mapconcat2(array_list)
    return samples;
end


function mlt_pdf(data, mu, cov)
    try 
        distribution = MvNormal(mu, Matrix(Hermitian(cov)));
        res = pdf(distribution, data);
        if isnan(res)
            return 1e6;
        end
        return res;
    catch
        return 1e-6;
    end
end

function k_means(k, tol, max_iter, data)
    centroids = data[rand(1:size(data)[1], k), :];
    for i = 1:max_iter
        classifications = Dict();
        for j = 1:k
            classifications[j] = [0 0];
        end
        for j = 1:length(data[:,1])
            distances = [];
            for jj = 1:k
                push!(distances, norm(data[j,:] .- centroids[jj, :]));
            end
            classifications[mapslices(argmin, distances, dims=1)[1]] = [classifications[mapslices(argmin, distances, dims=1)[1]]; reshape(data[j,:],(1,2))]; 
        end
        
        for j = 1:k
            classifications[j] = classifications[j][2:length(classifications[j][:,1]), :]
        end
        prev_centroids = centroids;
        
        for j = 1:k
            if (size(classifications[j])[1] != 0)
                centroids[j,:] = mean(classifications[j], dims = 1);
            end
        end
        
        optimized = true;
        for j in 1:length(centroids[:,1])
            original_centroid = prev_centroids[j];
            current_centroid = centroids[j];
            if abs(sum((current_centroid .- original_centroid) ./ original_centroid .*100.0)) > tol 
                    optimized = false; 
            end
        end
        if optimized
            break
        end
    end
    return centroids
end

function gmm_em(max_iterations, num_clusters, data)
    pi_ = ones(num_clusters)/num_clusters;
    mu = sort(k_means(num_clusters, 0.001, 200, data), dims = 1);
    for i = 1:2
        mu = mu .+ sort(k_means(num_clusters, 0.001, 200, data), dims = 1);
    end
    mu = mu ./ 3;
    cov = []
    for i = 1:num_clusters
        push!(cov, [[5. 0.]; [0. 5.]]);
    end
    ric = zeros((length(data[:,1]), num_clusters));
    reg_cov = 1e-6 .* [[5. 0.];[0. 5.]];
    log_likelihoods = [];
    for i = 1:max_iterations
        ric = zeros((length(data[:,1]), num_clusters));
        
        for j = 1:num_clusters
            covc = cov[j] .+ reg_cov;
            mn = [0];
            for num = 1:length(data[:,1])
                mn = [mn; [mlt_pdf(data[num, :], mu[j,:], covc)]];
            end
        
            mn = mn[2:length(mn)];
            ric[:, j] = pi_[j] .* mn;
        end
    
        for j = 1:length(data[:,1])
            ric[j, :] = ric[j, :] ./ sum(ric[j, :]);
        end
        
        mc = sum(ric, dims=1);
        pi_ = mc ./ sum(mc);
        mu = transpose((transpose(data) * ric) ./ [mc; mc]);
        
        cov = []
        for j = 1:num_clusters
            var1 = 1/mc[j] * dot(ric[:, j], (data[:,1] .- mu[j, 1]).^2);
            var2 = 1/mc[j] * dot(ric[:, j], (data[:,2] .- mu[j, 2]).^2);
            push!(cov, [[var1 0.]; [0. var2]]);
        end
    
        likelihood_sum = 0;
        for j = 1:num_clusters
            mn = [0];
            for num = 1:length(data[:,1])
                mn = [mn; [mlt_pdf(data[num,:], mu[j,:], cov[j] + reg_cov)]];
            end
            mn = mn[2:length(mn)];
            likelihood_sum = likelihood_sum .+ ric[:,j].*mn;
        end
        push!(log_likelihoods, sum(log.(likelihood_sum)));
        if i >= 20 && abs((log_likelihoods[i] - log_likelihoods[i-1])/log_likelihoods[i]) < 0.01
            break;
        end
    end
    
    likelihood = log_likelihoods[length(log_likelihoods)];
    bic = - 2*likelihood + log(size(data)[1])*(15*num_clusters);
    return pi_, mu, cov, ric, log_likelihoods, bic;
end

function main()
    # path = "/Users/clara/Desktop"
    # f_name = "data.csv"
    path = "D:/JI/2020 summer/VE414 Bayesian Analysis/Project/zzp";
    f_name = "data_proj_414.csv";
    data = generate_samples(path, f_name);
    println("----Samples are generated successfully----");
    max_iterations = 50;

    println("-----------Start training model-----------");
    bics = [0];
    for k=1:30
        pi_, mu, cov, ric, log_likelihoods, bic = gmm_em(max_iterations , k, data);
        pi_, mu, cov, ric, log_likelihoods, bic_ = gmm_em(max_iterations , k, data);
        bics = [bics;minimum([bic, bic_])];
        println("--Clusters: $(k)  BIC: $(mean([bic, bic_]))--");
    end
    bics = bics[2:length(bics[:,1]),:];
    println("-----------Stop training model------------");

    original_data = CSV.read(path*"/"*f_name);
    positions = [original_data["X"] original_data["Y"]];
    area_grid = zeros((107, 107));
    for i = 1:size(positions)[1]
        area_grid[Int32(positions[i, 1] - positions[i, 1] % 1) + 1, Int32(positions[i, 2] - positions[i, 2] % 1) + 1] = 1
    end
    area = sum(area_grid);
    lamb =  (mapslices(argmin,bics[:,1],dims=1)[1] * 107 * 107 / area);

    total = 0;
    mode_ = pdf(Poisson(lamb), 0);

    for i = 1:2*round(lamb)
        if pdf(Poisson(lamb), i) > mode_
            mode_ = pdf(Poisson(lamb), i);
            total = i;
        end
    end
    println("-------Total number of Jiulings $(total)------");
    return total;
end