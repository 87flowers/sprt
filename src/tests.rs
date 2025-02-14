use crate::gsprt::{PentanomialResult, SprtParameters};

#[test]
fn sprt_threshold_test() {
    let examples = [
        (485, 1923, 2942, 1937, 594, 0.0, 5.0, Some(true)),
        (261, 739, 2683, 737, 253, -10.0, 0.0, Some(true)),
        (63, 252, 385, 250, 74, 0.0, 10.0, None),
        (527, 1007, 1932, 933, 511, 0.0, 5.0, Some(false)),
        (175, 305, 694, 291, 157, 0.0, 10.0, Some(false)),
    ];
    for (ll, dl, wl_dd, wd, ww, elo0, elo1, expected_result) in examples {
        let penta = PentanomialResult {
            ll,
            dl,
            wl_dd,
            wd,
            ww,
        };
        let sprt = SprtParameters::new(elo0, elo1, 0.05, 0.10);
        let llr = sprt.llr(penta);
        let (lower_bound, upper_bound) = sprt.llr_bounds();
        let result: Option<bool> = if llr <= lower_bound {
            Some(false)
        } else if llr >= upper_bound {
            Some(true)
        } else {
            None
        };
        assert!(expected_result == result);
    }
}

#[test]
fn sprt_llr_test() {
    let examples = [
        (440, 2910, 5170, 2888, 455, 0.0, 5.0, -2.27),
        (142, 620, 1122, 699, 188, 0.0, 5.0, 2.99),
        (349, 1561, 3340, 1604, 359, -5.0, 0.0, 2.90),
        (98, 382, 674, 369, 71, -5.0, 0.0, -1.11),
    ];
    for (ll, dl, wl_dd, wd, ww, elo0, elo1, expected_llr) in examples {
        let penta = PentanomialResult {
            ll,
            dl,
            wl_dd,
            wd,
            ww,
        };
        let sprt = SprtParameters::new(elo0, elo1, 0.05, 0.10);
        let llr = sprt.llr(penta);
        let error = f64::abs(llr - expected_llr);
        assert!(error <= 0.005);
    }
}

#[test]
fn extreme_values_test() {
    let examples = [
        (0, 0, 0, 0, 0, 0.0, 5.0),
        (1, 0, 0, 0, 0, 0.0, 5.0),
        (0, 1, 0, 0, 0, 0.0, 5.0),
        (0, 0, 1, 0, 0, 0.0, 5.0),
        (0, 0, 0, 1, 0, 0.0, 5.0),
        (0, 0, 0, 0, 1, 0.0, 5.0),
        (10, 0, 0, 0, 0, 0.0, 5.0),
        (0, 10, 0, 0, 0, 0.0, 5.0),
        (0, 0, 10, 0, 0, 0.0, 5.0),
        (0, 0, 0, 10, 0, 0.0, 5.0),
        (0, 0, 0, 0, 10, 0.0, 5.0),
        (100, 0, 0, 0, 0, 0.0, 5.0),
        (0, 100, 0, 0, 0, 0.0, 5.0),
        (0, 0, 100, 0, 0, 0.0, 5.0),
        (0, 0, 0, 100, 0, 0.0, 5.0),
        (0, 0, 0, 0, 100, 0.0, 5.0),
        (1000, 0, 0, 0, 0, 0.0, 5.0),
        (0, 1000, 0, 0, 0, 0.0, 5.0),
        (0, 0, 1000, 0, 0, 0.0, 5.0),
        (0, 0, 0, 1000, 0, 0.0, 5.0),
        (0, 0, 0, 0, 1000, 0.0, 5.0),
        (10000, 0, 0, 0, 0, 0.0, 5.0),
        (0, 10000, 0, 0, 0, 0.0, 5.0),
        (0, 0, 10000, 0, 0, 0.0, 5.0),
        (0, 0, 0, 10000, 0, 0.0, 5.0),
        (0, 0, 0, 0, 10000, 0.0, 5.0),
        (100000, 0, 0, 0, 0, 0.0, 5.0),
        (0, 100000, 0, 0, 0, 0.0, 5.0),
        (0, 0, 100000, 0, 0, 0.0, 5.0),
        (0, 0, 0, 100000, 0, 0.0, 5.0),
        (0, 0, 0, 0, 100000, 0.0, 5.0),
        (100000, 100000, 0, 0, 0, 0.0, 5.0),
        (100000, 0, 100000, 0, 0, 0.0, 5.0),
        (100000, 0, 0, 100000, 0, 0.0, 5.0),
        (100000, 0, 0, 0, 100000, 0.0, 5.0),
        (0, 100000, 100000, 0, 0, 0.0, 5.0),
        (0, 100000, 0, 100000, 0, 0.0, 5.0),
        (0, 100000, 0, 0, 100000, 0.0, 5.0),
        (0, 0, 100000, 100000, 0, 0.0, 5.0),
        (0, 0, 100000, 0, 100000, 0.0, 5.0),
        (0, 0, 0, 100000, 100000, 0.0, 5.0),
        (100000, 100000, 100000, 0, 0, 0.0, 5.0),
        (100000, 100000, 0, 100000, 0, 0.0, 5.0),
        (100000, 100000, 0, 0, 100000, 0.0, 5.0),
        (0, 100000, 100000, 100000, 0, 0.0, 5.0),
        (0, 100000, 100000, 0, 100000, 0.0, 5.0),
        (0, 0, 100000, 100000, 100000, 0.0, 5.0),
        (0, 100000, 100000, 100000, 100000, 0.0, 5.0),
        (100000, 0, 100000, 100000, 100000, 0.0, 5.0),
        (100000, 100000, 0, 100000, 100000, 0.0, 5.0),
        (100000, 100000, 100000, 0, 100000, 0.0, 5.0),
        (100000, 100000, 100000, 100000, 0, 0.0, 5.0),
    ];
    for example in examples {
        let (ll, dl, wl_dd, wd, ww, elo0, elo1) = example;
        let penta = PentanomialResult {
            ll,
            dl,
            wl_dd,
            wd,
            ww,
        };
        let sprt = SprtParameters::new(elo0, elo1, 0.05, 0.10);
        let llr = sprt.llr(penta);
        println!("{:?} = {}", example, llr);
    }
}

#[test]
fn random_values_test() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..1000000 {
        let penta = PentanomialResult {
            ll: rng.gen_range(0..100000),
            dl: rng.gen_range(0..100000),
            wl_dd: rng.gen_range(0..100000),
            wd: rng.gen_range(0..100000),
            ww: rng.gen_range(0..100000),
        };
        let sprt = SprtParameters::new(0.0, 5.0, 0.05, 0.05);
        _ = sprt.llr(penta);
    }
}
