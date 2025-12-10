ABSTRACT

This research explores the effectiveness of an adapted form of adversarial training as a
defence against Poison Backdoor attacks on Convolutional Neural Networks (CNNs) trained
on the MNIST dataset. The study initially focused on various perceptible backdoor triggers,
introducing the pertubation to both attacking and defensive samples, differenciated by their
labelling (bad/clean). Perceptible triggers leverage high-level features such as shape and
color, but these attacks proved resilient during adversarial training against all pertubatutions
but a replica of themselves. Subsequent experiments shifted to imperceptible triggers,
specifically those generated using Frequency Domain Manipulations (FDM), which rely on
subtle, low-level features. The results demonstrated that adversarial training with various
adversarial pertubations in the frequency domain can effectively mitigate the FDM backdoor
attacks, desensitising the model to the imperceptible trigger.
The research draws on a synthesis of existing defence strategies, including a systematic
unlearning method (Wang et al., 2019) and an advanced adversarial training technique
(Geiping et al., 2021). This combination aimed to build robustness against both perceptible
and imperceptible backdoor triggers. The results showed success in some areas and
highlighted the need for further exploration into the perturbation budget of the poison trigger
in comparison to the defensive one.
