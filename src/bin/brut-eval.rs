//! `brut-eval`: the hidden evaluator process.
//!
//! Usage: brut-eval --secret <file> --budget <n> --step <f>
//! Reads one request JSON from stdin (at most 1 MiB) and writes one response JSON to stdout:
//! - no "kind" key: legacy MLP `EvalRequest` -> `EvalResponse` (unchanged). Its model file
//!   holds an MLP or a CNN (`model::Model`);
//! - "kind": "learner": `LearnerEvalRequest` -> `LearnerEvalResponse`;
//! - "kind": "arc": `ArcEvalRequest` -> `ArcEvalResponse`.
//!
//! Exit code 0 on success, 2 on usage error, 3 on evaluation error.
//! Error messages never quote request content, hidden data or the secret.
#![forbid(unsafe_code)]

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

use brutmaschine::hidden_eval::{AnyEvalRequest, HiddenEvaluator, read_limited};
use brutmaschine::learner_hidden::LearnerEvaluator;

/// Largest request accepted on stdin.
const MAX_REQUEST_BYTES: u64 = 1024 * 1024;

const USAGE: &str = "usage: brut-eval --secret <file> --budget <n> --step <f>";

struct Args {
    secret: PathBuf,
    budget: usize,
    step: f64,
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut secret: Option<PathBuf> = None;
    let mut budget: Option<usize> = None;
    let mut step: Option<f64> = None;
    let mut i = 0;

    while i < raw.len() {
        let flag = raw[i].as_str();
        let Some(value) = raw.get(i + 1) else {
            return Err(format!("missing value for {flag}"));
        };

        match flag {
            "--secret" if secret.is_none() => secret = Some(PathBuf::from(value)),
            "--budget" if budget.is_none() => {
                budget = Some(value.parse().map_err(|_| format!("bad --budget {value}"))?);
            }
            "--step" if step.is_none() => {
                step = Some(value.parse().map_err(|_| format!("bad --step {value}"))?);
            }
            _ => return Err(format!("unexpected or repeated argument {flag}")),
        }

        i += 2;
    }

    match (secret, budget, step) {
        (Some(secret), Some(budget), Some(step)) => Ok(Args {
            secret,
            budget,
            step,
        }),
        _ => Err("--secret, --budget and --step are required".to_string()),
    }
}

/// Read one request from `input`, evaluate it, return the response JSON (or the message).
fn respond(args: &Args, input: impl Read) -> Result<String, String> {
    let text = read_limited(input, MAX_REQUEST_BYTES).map_err(|e| e.to_string())?;
    let req = AnyEvalRequest::from_json(&text).map_err(|e| e.to_string())?;

    let json = match req {
        AnyEvalRequest::Mlp(req) => {
            let mut evaluator = HiddenEvaluator::open(&args.secret, args.budget, args.step)
                .map_err(|e| e.to_string())?;
            let resp = evaluator.evaluate(&req).map_err(|e| e.to_string())?;

            serde_json::to_string(&resp)
        }
        AnyEvalRequest::Learner(req) => {
            let mut evaluator = LearnerEvaluator::open(&args.secret, args.budget, args.step)
                .map_err(|e| e.to_string())?;
            let resp = evaluator.evaluate(&req).map_err(|e| e.to_string())?;

            serde_json::to_string(&resp)
        }
        AnyEvalRequest::Arc(req) => {
            let mut evaluator = LearnerEvaluator::open(&args.secret, args.budget, args.step)
                .map_err(|e| e.to_string())?;
            let resp = evaluator.evaluate_arc(&req).map_err(|e| e.to_string())?;

            serde_json::to_string(&resp)
        }
    };

    json.map_err(|e| e.to_string())
}

/// Replace the default panic message: its payload could quote hidden data (for example a
/// task seed from library code). Only the source location is printed.
fn install_quiet_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let place = info
            .location()
            .map(|l| format!(" at {}:{}", l.file(), l.line()))
            .unwrap_or_default();

        eprintln!("brut-eval: internal error{place} (message suppressed)");
    }));
}

fn main() -> ExitCode {
    install_quiet_panic_hook();

    let raw: Vec<String> = std::env::args().skip(1).collect();

    let args = match parse_args(&raw) {
        Ok(a) => a,
        Err(m) => {
            eprintln!("brut-eval: {m}\n{USAGE}");

            return ExitCode::from(2);
        }
    };

    // A panic (also one inside a rayon worker) ends as an evaluation error, exit code 3.
    let answer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        respond(&args, std::io::stdin().lock())
    }));

    match answer {
        Ok(Ok(json)) => {
            println!("{json}");

            ExitCode::SUCCESS
        }
        Ok(Err(m)) => {
            eprintln!("brut-eval: {m}");

            ExitCode::from(3)
        }
        Err(_) => ExitCode::from(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_all_three_flags() {
        let a = parse_args(&strings(&[
            "--secret", "s.env", "--budget", "5", "--step", "0.01",
        ]))
        .expect("valid");

        assert_eq!(a.secret, PathBuf::from("s.env"));
        assert_eq!(a.budget, 5);
        assert_eq!(a.step, 0.01);
    }

    #[test]
    fn rejects_bad_args() {
        assert!(parse_args(&strings(&[])).is_err());
        assert!(
            parse_args(&strings(&[
                "--secret", "s.env", "--budget", "x", "--step", "0.1"
            ]))
            .is_err()
        );
        assert!(
            parse_args(&strings(&[
                "--secret", "a", "--secret", "b", "--budget", "1", "--step", "0"
            ]))
            .is_err()
        );
        assert!(parse_args(&strings(&["--secret", "s.env", "--budget", "1", "--step"])).is_err());
        assert!(parse_args(&strings(&["--other", "1"])).is_err());
    }

    /// A distinctive seed: stdout and stderr of brut-eval must never contain it.
    const SEED: u64 = 13_579_246_801_357_924_680;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brut-eval-bin-{name}-{}", std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        set_mode(&dir, 0o700);

        dir
    }

    fn set_mode(path: &std::path::Path, mode: u32) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).expect("chmod");
        }

        #[cfg(not(unix))]
        let _ = (path, mode);
    }

    fn write_private(path: &std::path::Path, text: &str) {
        std::fs::write(path, text).expect("write");
        set_mode(path, 0o600);
    }

    fn args(secret: &std::path::Path, budget: usize) -> Args {
        Args {
            secret: secret.to_path_buf(),
            budget,
            step: 0.01,
        }
    }

    /// What main would print: stdout on success, stderr on failure.
    fn printed(r: &Result<String, String>) -> String {
        match r {
            Ok(out) => out.clone(),
            Err(err) => err.clone(),
        }
    }

    fn assert_no_secret(text: &str) {
        assert!(!text.contains(&SEED.to_string()), "secret leaked: {text}");
        assert!(
            !text.contains(&format!("{SEED:x}")),
            "secret leaked: {text}"
        );
    }

    fn learner_json(run_id: &str, tier: &str) -> String {
        format!(
            "{{\"kind\":\"learner\",\"run_id\":\"{run_id}\",\"candidate_id\":\"3fa1c2d4e5f60718\",\
             \"protocol_hash\":\"{ph}\",\"tier\":\"{tier}\",\
             \"program\":{{\"setup\":[],\"predict\":[],\"learn\":[]}}}}",
            ph = "b".repeat(64)
        )
    }

    #[test]
    fn oversized_and_malformed_requests_are_refused_without_quoting_them() {
        let a = args(std::path::Path::new("/nonexistent/hidden.env"), 1);
        let big = vec![b' '; (MAX_REQUEST_BYTES + 1) as usize];
        let err = respond(&a, &big[..]).expect_err("too big");

        assert!(err.contains("larger than"), "{err}");

        for text in [
            "{\"kind\":\"zz-private\"}".to_string(),
            "[1,2,3]".to_string(),
            "zz-private not json".to_string(),
            learner_json("zz-private", "hidden_grammar").replace("\"tier\"", "\"zz-private\""),
        ] {
            let err = respond(&a, text.as_bytes()).expect_err("refused");

            assert!(!err.contains("zz-private"), "{err}");
        }
    }

    #[test]
    fn mlp_requests_work_as_before_and_never_print_the_secret() {
        use brutmaschine::genome::Activation;
        use brutmaschine::hidden_eval::{EvalRequest, EvalResponse, HIDDEN_SEED_KEY};
        use brutmaschine::nn::Mlp;
        use brutmaschine::task::TaskKind;

        let dir = tmp_dir("mlp");
        let secret_dir = dir.join("secret");
        let models = dir.join("models");

        std::fs::create_dir_all(&secret_dir).expect("secret dir");
        std::fs::create_dir_all(&models).expect("models dir");
        set_mode(&secret_dir, 0o700);

        let secret = secret_dir.join("hidden.env");

        write_private(&secret, &format!("{HIDDEN_SEED_KEY}={SEED}\n"));

        let model = Mlp {
            sizes: vec![2, 4, 2],
            activation: Activation::Tanh,
            weights: vec![vec![0.1; 8], vec![0.1; 8]],
            biases: vec![vec![0.0; 4], vec![0.0; 2]],
        };
        let model_path = models.join("m.json");

        std::fs::write(&model_path, serde_json::to_vec(&model).expect("json")).expect("model");

        let req = serde_json::to_string(&EvalRequest {
            candidate_id: "abc".into(),
            model_path,
            task: TaskKind::TwoSpirals,
        })
        .expect("json");

        let first = respond(&args(&secret, 1), req.as_bytes());
        let resp: EvalResponse =
            serde_json::from_str(first.as_ref().expect("scored")).expect("response");

        assert_eq!(resp.candidate_id, "abc");
        assert_eq!((resp.queries_used, resp.queries_left), (1, 0));
        assert_no_secret(&printed(&first));

        // Budget exhausted.
        let second = respond(&args(&secret, 1), req.as_bytes());

        assert!(second.is_err());
        assert_no_secret(&printed(&second));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn learner_and_arc_requests_never_print_the_secret() {
        use brutmaschine::learner_hidden::{BUDGET_TOTAL_KEY, LEARNER_SEED_KEY};

        let dir = tmp_dir("learner");
        let secret = dir.join("hidden.env");
        let good_run = "a".repeat(64);
        let cases: Vec<(String, String)> = vec![
            // Malformed run id, public tier, ARC without sources: refused before any scoring.
            (
                format!("{LEARNER_SEED_KEY}={SEED}\n"),
                learner_json("xyz", "hidden_grammar"),
            ),
            (
                format!("{LEARNER_SEED_KEY}={SEED}\n"),
                learner_json(&good_run, "search"),
            ),
            (
                format!("{LEARNER_SEED_KEY}={SEED}\n"),
                learner_json(&good_run, "hidden_grammar")
                    .replace("\"kind\":\"learner\"", "\"kind\":\"arc\"")
                    .replace(
                        "\"tier\":\"hidden_grammar\"",
                        "\"set\":\"a1_eval\",\"train_sources\":[]",
                    ),
            ),
            // Broken secret files.
            (
                format!("{LEARNER_SEED_KEY}={SEED}x\n"),
                learner_json(&good_run, "hidden_grammar"),
            ),
            (
                format!("{LEARNER_SEED_KEY}={SEED}\n{BUDGET_TOTAL_KEY}=many\n"),
                learner_json(&good_run, "hidden_grammar"),
            ),
            (
                format!("BRUT_HIDDEN_SEED={SEED}\n"),
                learner_json(&good_run, "hidden_grammar"),
            ),
        ];

        for (secret_text, request) in cases {
            write_private(&secret, &secret_text);

            let r = respond(&args(&secret, 12), request.as_bytes());

            assert!(r.is_err(), "{request}");
            assert_no_secret(&printed(&r));
        }

        // A secret readable by others is refused, and the refusal does not print it.
        write_private(&secret, &format!("{LEARNER_SEED_KEY}={SEED}\n"));
        set_mode(&secret, 0o644);

        let r = respond(
            &args(&secret, 12),
            learner_json(&good_run, "hidden_grammar").as_bytes(),
        );

        assert!(r.is_err());
        assert_no_secret(&printed(&r));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
