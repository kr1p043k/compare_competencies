import { Card, CardBody, CardHeader } from "../../../design";
import type { AccountProps } from "./types";

export function AccountCard({ displayName, email, roleLabel }: AccountProps & { roleLabel: string }) {
  return (
    <Card>
      <CardHeader title="Учётная запись" description={roleLabel} />
      <CardBody className="space-y-1 text-sm text-foreground">
        <p>
          <span className="text-muted-foreground">ФИО: </span>
          {displayName || "-"}
        </p>
        <p>
          <span className="text-muted-foreground">Email: </span>
          {email || "-"}
        </p>
      </CardBody>
    </Card>
  );
}
