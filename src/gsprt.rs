use bacon_sci::roots::itp;

/// Pentanomial results considers the result of paired games. Games are generally
/// played in pairs with the same opening but with reversed colors in order to
/// reduce opening bias. Since there are correlations between paired games, the
/// pentanomial model a more accurate model. Notably, it can be used to better
/// estimate the variance in match score than a trinominal model can.
///
/// Taking a loss as a 0, a win as a 1 and a draw as a 1/2, there are five possible
/// outcomes for a pair of games:
///
///    0  : Two losses
///   1/2 : A loss and a draw
///    1  : Two draws, or a win and a loss
///   3/2 : A win and a draw
///    2  : Two wins
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PentanomialResult {
    pub ll: usize,
    pub dl: usize,
    pub wl_dd: usize,
    pub wd: usize,
    pub ww: usize,
}

impl PentanomialResult {
    /// Calculate an empirical probability distribution given a set of observed results.
    pub fn to_pd(self) -> ProbabilityDistribution<5> {
        fn regularize(value: usize) -> f64 {
            if value == 0 { 1e-3 } else { value as f64 }
        }
        let ll = regularize(self.ll);
        let dl = regularize(self.dl);
        let wl_dd = regularize(self.wl_dd);
        let wd = regularize(self.wd);
        let ww = regularize(self.ww);
        let game_count = ll + dl + wl_dd + wd + ww;
        ProbabilityDistribution::<5> {
            game_count,
            score: [0.0, 0.25, 0.5, 0.75, 1.0],
            prob: [
                ll / game_count,
                dl / game_count,
                wl_dd / game_count,
                wd / game_count,
                ww / game_count,
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbabilityDistribution<const N: usize> {
    pub game_count: f64,
    pub score: [f64; N],
    pub prob: [f64; N],
}

fn mean<const N: usize>(x: [f64; N], p: [f64; N]) -> f64 {
    (0..N).map(|i| p[i] * x[i]).sum()
}

fn mean_and_variance<const N: usize>(x: [f64; N], p: [f64; N]) -> (f64, f64) {
    let mu = mean(x, p);
    (mu, (0..N).map(|i| p[i] * (x[i] - mu).powi(2)).sum())
}

impl<const N: usize> ProbabilityDistribution<N> {
    /// Compute log-likelihood ratio for t = t0 versus t = t1.
    fn llr(&self, t0: f64, t1: f64) -> f64 {
        let p0 = self.mle(0.5, t0);
        let p1 = self.mle(0.5, t1);
        self.game_count * mean(core::array::from_fn(|i| p1[i].ln() - p0[i].ln()), self.prob)
    }

    /// Compute the maximum likelihood estimate for a discrete
    /// probability distribution that has t = (mu - mu_ref) / sigma,
    /// given `self` is an empirical distribution.
    ///
    /// See section 4.1 of [1] for details.
    fn mle(&self, mu_ref: f64, t_star: f64) -> [f64; N] {
        let theta_epsilon = 1e-7;
        let mle_epsilon = 1e-4;

        // This is an iterative method, so we need to start with
        // an initial value. As suggested in [1], we start with a
        // uniform distribution.
        let mut p = [1.0 / N as f64; N];

        loop {
            // Store our current estimate away to detect convegence.
            let prev_p = p;

            // Calculate phi.
            let (mu, variance) = mean_and_variance(self.score, p);
            let phi: [f64; N] = core::array::from_fn(|i| {
                let a_i = self.score[i];
                let sigma = variance.sqrt();
                a_i - mu_ref - 0.5 * t_star * sigma * (1.0 + ((a_i - mu) / sigma).powi(2))
            });

            // We need to find a subset of the possible solutions for theta,
            // so we need to calculate our constraints for theta.
            let u = phi
                .iter()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).expect("unexpected NaN"))
                .unwrap();
            let v = phi
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).expect("unexpected NaN"))
                .unwrap();
            let min_theta = -1.0 / v;
            let max_theta = -1.0 / u;

            // Solve equation 4.9 in [1] for theta.
            let theta = itp(
                (min_theta, max_theta),
                |x: f64| {
                    (0..N)
                        .map(|i| self.prob[i] * phi[i] / (1.0 + x * phi[i]))
                        .sum()
                },
                0.1,
                1.5,
                0.99,
                theta_epsilon,
            )
            .unwrap();

            // Calculate new estimate
            p = core::array::from_fn(|i| {
                let phat_i = self.prob[i];
                phat_i / (1.0 + theta * phi[i])
            });

            // Good enough?
            if (0..N).all(|i| (prev_p[i] - p[i]).abs() < mle_epsilon) {
                break;
            }
        }

        p
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SprtParameters {
    lower_bound: f64,
    upper_bound: f64,
    nelo0: f64,
    nelo1: f64,
    t0: f64,
    t1: f64,
}

impl SprtParameters {
    /// Constructs parameters to use for a SPRT test.
    /// nelo0 : Represents the H0 hypothesis that the normalized elo difference is nelo0
    /// nelo1 : Represents the H1 hypothesis that the normalized elo difference is nelo1
    /// alpha : False positive rate (Type I error)
    /// beta : False negative rate (Type II error)
    pub fn new(nelo0: f64, nelo1: f64, alpha: f64, beta: f64) -> SprtParameters {
        let c_et = 800.0 / f64::ln(10.0);
        let lower_bound = f64::ln(beta / (1.0 - alpha));
        let upper_bound = f64::ln((1.0 - beta) / alpha);
        let t0 = nelo0 / c_et;
        let t1 = nelo1 / c_et;
        SprtParameters {
            lower_bound,
            upper_bound,
            nelo0,
            nelo1,
            t0,
            t1,
        }
    }

    /// Bounds on LLR for SPRT termination.
    /// If LLR falls below the lower bound, that demonstrates the hypothesis that elo = elo0 is more likely.
    /// If LLR falls above the upper bound, that demonstrates the hypothesis that elo = elo1 is more likely,
    /// If LLR falls within these bounds, more data is required.
    pub fn llr_bounds(self: SprtParameters) -> (f64, f64) {
        (self.lower_bound, self.upper_bound)
    }

    /// Returns the elo bounds provided to the constructor
    pub fn nelo_bounds(self: SprtParameters) -> (f64, f64) {
        (self.nelo0, self.nelo1)
    }

    /// Calculates the LLR for the given pentanomial results, given our SPRT parameters
    pub fn llr(self: SprtParameters, penta: PentanomialResult) -> f64 {
        let pd = penta.to_pd();
        pd.llr(self.t0 * f64::sqrt(2.0), self.t1 * f64::sqrt(2.0))
    }
}
