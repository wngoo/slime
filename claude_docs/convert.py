else:
    tensors = state_dict[key]
    first = tensors[0]

    if isinstance(first, torch.Tensor) and first.dim() == 0:
        values = []
        for t in tensors:
            if not isinstance(t, torch.Tensor) or t.dim() != 0:
                raise TypeError(
                    f"Expected all scalar tensors for key={key}, got: "
                    f"{[type(x) for x in tensors]}"
                )
            values.append(t.item())

        if not all(torch.equal(first, t) for t in tensors[1:]):
            raise ValueError(
                f"Scalar tensor differs across ranks for key={key}: {values}"
            )

        print(f"[merge scalar] keep one copy: key={key}, value={first.item()}")
        state_dict[key] = first.contiguous()

    else:
        print(
            f"[merge concat] key={key}, "
            f"shape0={tuple(first.shape) if isinstance(first, torch.Tensor) else type(first)}"
        )
        state_dict[key] = torch.cat(tensors, dim=0)
