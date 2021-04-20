using DrMZ
using Documenter

DocMeta.setdocmeta!(DrMZ, :DocTestSetup, :(using DrMZ); recursive=true)

makedocs(;
    modules=[DrMZ],
    authors="Brek Meuris",
    repo="https://github.com/brekmeuris/DrMZ.jl/blob/{commit}{path}#{line}",
    sitename="DrMZ.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://brekmeuris.github.io/DrMZ.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/brekmeuris/DrMZ.jl",
    devbranch="main",
    branch="gh-pages",
)
