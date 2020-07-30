using ScatterNNlib
using Documenter

makedocs(;
    modules=[ScatterNNlib],
    authors="Yueh-Hua Tu",
    repo="https://github.com/yuehhua/ScatterNNlib.jl/blob/{commit}{path}#L{line}",
    sitename="ScatterNNlib.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yuehhua.github.io/ScatterNNlib.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yuehhua/ScatterNNlib.jl",
)
