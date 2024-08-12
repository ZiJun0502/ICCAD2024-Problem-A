//setup mode

set system mode setup

set log file lec.log -replace

setenv TOP_DESIGN top_809568696_809776567_809698999_863110837_1234615


read design ../release/netlists/design4.v -golden -verilog2k
read design ../playground/design_preprocessed.v -revised

set root module $TOP_DESIGN -golden
set root module $TOP_DESIGN -revised

