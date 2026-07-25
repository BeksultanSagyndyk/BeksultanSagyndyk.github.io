# frozen_string_literal: true

source "https://rubygems.org"

gemspec

# ffi 1.17+ requires Ruby >= 3.0, which the jekyll/builder:3.8 CI image and
# older local Rubies can't run. 1.16.3 works on both Ruby 2.6 and 3.x.
# See the committed Gemfile.lock, which pins the exact versions.
gem "ffi", "< 1.17"

