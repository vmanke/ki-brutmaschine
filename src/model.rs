//! A trained model: an MLP or a CNN, and its file format.
//!
//! `evolve` saves each trained model of status Ok as `models/<id>.json`, and `brut-eval`
//! reads it back. An MLP file is the bare JSON of `nn::Mlp`, as before CNNs. A CNN file is
//! the JSON of `cnn::model::CnnModel`, which starts with `"kind":"cnn"`. A file without a
//! `"kind"` key is an MLP file, as for the requests of `brut-eval` (`AnyEvalRequest`).

use serde::{Deserialize, Serialize, Serializer};

use crate::cnn::model::CnnModel;
use crate::nn::Mlp;
use crate::util::{BrutError, Result};

/// The model that `train` returns and `score::evaluate` scores.
#[derive(Clone, Debug, PartialEq)]
pub enum Model {
    Mlp(Mlp),
    Cnn(CnnModel),
}

impl Serialize for Model {
    /// An MLP as the bare JSON of `nn::Mlp`, so its file keeps its bytes. A CNN as the
    /// JSON of `CnnModel`, which carries `"kind":"cnn"`.
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Model::Mlp(mlp) => mlp.serialize(serializer),
            Model::Cnn(cnn) => cnn.serialize(serializer),
        }
    }
}

/// The only value of the `"kind"` key a model file may have.
#[derive(Deserialize)]
enum Kind {
    #[serde(rename = "cnn")]
    Cnn,
}

/// Reads the `"kind"` key of a model file and skips every other key.
#[derive(Deserialize)]
struct Probe {
    kind: Option<Kind>,
}

impl Model {
    /// Parse a model file. Without a `"kind"` key it is an MLP; with `"kind":"cnn"` a CNN.
    ///
    /// The errors are generic: a parser's message could quote strings from the file. A CNN
    /// file holds at most `config::HARD_MAX_PARAMS` trained values and at most
    /// `cnn::model::MAX_RUNNING_VALUES` running statistics (see `CnnModel`), so a hostile
    /// file cannot make the parser allocate more for them. The values are not checked
    /// against a task here; `CnnModel::check` does that.
    pub fn from_json(bytes: &[u8]) -> Result<Model> {
        let invalid = |what: &str| BrutError::Invalid(format!("model file is not {what}"));

        let probe: Probe = serde_json::from_slice(bytes).map_err(|_| {
            invalid("a valid model (JSON of nn::Mlp, or of cnn::CnnModel with \"kind\": \"cnn\")")
        })?;

        match probe.kind {
            None => serde_json::from_slice(bytes)
                .map(Model::Mlp)
                .map_err(|_| invalid("a valid model (JSON of nn::Mlp)")),
            Some(Kind::Cnn) => serde_json::from_slice(bytes)
                .map(Model::Cnn)
                .map_err(|_| invalid("a valid CNN model (JSON of cnn::CnnModel)")),
        }
    }

    /// Trainable values: weights and biases, and for a CNN also the scales and shifts of
    /// its norms.
    pub fn param_count(&self) -> usize {
        match self {
            Model::Mlp(mlp) => mlp.param_count(),
            Model::Cnn(cnn) => cnn.param_count(),
        }
    }

    /// True if every stored value is finite.
    pub fn is_finite(&self) -> bool {
        match self {
            Model::Mlp(mlp) => mlp.is_finite(),
            Model::Cnn(cnn) => cnn.is_finite(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnn::genome::testing;
    use crate::config::Limits;
    use crate::genome::Activation;
    use crate::task::{Split, TaskKind, generate};
    use crate::train;

    fn mlp() -> Mlp {
        Mlp {
            sizes: vec![2, 3, 2],
            activation: Activation::Tanh,
            weights: vec![vec![0.5; 6], vec![-0.25; 6]],
            biases: vec![vec![0.0; 3], vec![0.125; 2]],
        }
    }

    #[test]
    fn an_mlp_file_is_the_bare_mlp_json() {
        let model = Model::Mlp(mlp());
        let bytes = serde_json::to_vec(&model).expect("json");

        assert_eq!(bytes, serde_json::to_vec(&mlp()).expect("json"));
        assert_eq!(Model::from_json(&bytes).expect("parse"), model);
    }

    #[test]
    fn a_cnn_file_starts_with_its_kind_and_parses_back_equal() {
        let spec = TaskKind::Shapes.spec();
        let data = crate::cnn::train::testing::head(&generate(&spec, Split::Train, 1), 64);
        let genome = testing::vgg();
        let (model, _) = train::train(&genome, &spec, &data, &Limits::default()).expect("train");
        let text = serde_json::to_string(&model).expect("json");

        assert!(
            text.starts_with("{\"kind\":\"cnn\",\"arch\":{"),
            "{}",
            &text[..40]
        );
        assert_eq!(Model::from_json(text.as_bytes()).expect("parse"), model);
    }

    #[test]
    fn broken_files_give_generic_errors() {
        for text in [
            "",
            "[1, 2]",
            "{\"kind\": \"zz-private\"}",
            "{\"kind\": 7}",
            "{\"kind\": \"cnn\"}",
            "{\"sizes\": \"zz-private\"}",
            "{\"kind\": \"cnn\", \"zz-private\": 1}",
        ] {
            let message = Model::from_json(text.as_bytes())
                .expect_err("refused")
                .to_string();

            assert!(message.contains("model file is not"), "{message}");
            assert!(!message.contains("zz-private"), "{message}");
        }
    }
}
