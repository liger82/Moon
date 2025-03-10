# coding: utf-8

Gem::Specification.new do |spec|
  spec.name          = "jekyll-theme-type"
  spec.version       = "1.1"
  spec.authors       = ["Stuart"]
  spec.email         = ["xkdldhs82@naver.com"]

  spec.summary       = %q{A free and open-source Jekyll theme. Great for blogs and easy to customize.}
  spec.homepage      = "https://github.com/rohanchandra/type-theme"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^(assets|pages|_(includes|layouts|sass)/|(LICENSE|README|search.html)((\.(txt|md|markdown)|$)))}i)
  end

  spec.add_runtime_dependency "jekyll", "~> 3.4"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-category-pages", "~>1.0.0"

  spec.add_development_dependency "bundler", "~> 1.12"
  spec.add_development_dependency "rake", "~> 10.0"
end
