# AudioSample Style Guide

This document aims to provide a guide to how Rust code is written in Rust.
While we all aim to adhere Rust best practices sometimes, in the interest of the crate, we deviate.
Of course we should avoid unwrap/expect in library code, as a library should defer that to the user.
However, there are times where it can be justified as the best thing to do, as it signals something is fundamentally wrong.
There are other small things like this that this document discusses.
It also discusses how the the library **should** be extended, i.e. an architecture pattern exists and should be followed.
See [ARCHICTECURE.md] for a thorough explanation.
