include("utils.jl")

# For Gaussian
x1 = JLD2.load("../test-data-2D/UM-1817-shepplogan.jld2")["x1"]
# For masked Fourier
x2 = Float32.(FileIO.load("../test-data-2D/Mag13K.tif"))
x2 = x2[51:306, 651:906]'
x2 /= maximum(x2)
# For canonical Fourier
xall = Float32.(Gray.(FileIO.load("../test-data-2D/test-image-w-ref.png")))
x3 = imresize(xall[5:1114,5:1114], 256,256)
x3 /= maximum(x3)
zero_x3 = imresize(xall[5:1114, 1121:2230], 256,256)
ref_x3 = imresize(xall[5:1114, 2241:3350],256,256)
# For empirical transmission matrix
x4 = matread("../AmpSLM_16x16/myxtrue_final.mat")["xtrue"]
x4 /= maximum(abs.(xtrue))
# x4 = Float32.(Gray.(FileIO.load("../test-data-2D/logo.jpeg")))
# x4 = x4[:,1:256]'
# x4 = imresize(x4, (64,64), method=BSpline(Linear()))
jim(x1, xlim = (1, size(x1, 1)))
savefig("../result/xtrue_x1.pdf")
jim(x2, xlim = (1, size(x2, 1)))
savefig("../result/xtrue_x2.pdf")
jim(x3, xlim = (1, size(x3, 1)))
savefig("../result/xtrue_x3.pdf")
jim(ref_x3, xlim = (1, size(ref_x3, 1)))
savefig("../result/xtrue_x3_ref.pdf")
jim(x4, xlim = (1, size(x4, 1)))
savefig("../result/xtrue_x4.pdf")
jim(real(x4), xlim = (1, size(x4, 1)))
savefig("../result/xtrue_x4_real.pdf")
jim(imag(x4), xlim = (1, size(x4, 1)))
savefig("../result/xtrue_x4_imag.pdf")

jim(jim(x1), jim(x2), jim(x3), jim(x4))
JLD2.save("../test-data-2D/4-test-images.jld2",
          Dict("x1"=>x1,
          "x2"=>x2,
          "x3"=>x3,
          "zero_x3"=>zero_x3,
          "ref_x3"=>ref_x3,
          "x4"=>x4))
