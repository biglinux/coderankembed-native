#![no_main]
libfuzzer_sys::fuzz_target!(|data: &[u8]| coderankembed_native::fuzzing::tokenize(data));
