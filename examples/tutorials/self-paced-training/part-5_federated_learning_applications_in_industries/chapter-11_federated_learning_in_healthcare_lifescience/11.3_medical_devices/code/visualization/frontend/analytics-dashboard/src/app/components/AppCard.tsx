/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use client';
import { Card} from "@kui-react/card";
import { Flex } from "@kui-react/flex";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { AppNames, IconNames } from "../../../src/config";
import Image from "next/image";

interface AppCardProps {
    title: string;
}

export default function AppCard({ title }: AppCardProps) {
    const router = useRouter();
    const searchParams = useSearchParams();
    const pathName = usePathname();
    const [isReady, setIsReady] = useState(false);
    const handleClick = (id: string) => {
        router.push(`/stats/${id}`);
        if (isReady) {
            console.log("is ready!!");
            router?.push(`/stats/${id}`);
        } else {
            console.log("is not ready!!");
        }
    }

    useEffect(() => {
        if (searchParams && pathName) {
            setIsReady(true);
        }
    }, [searchParams, pathName]);

    const iconName = IconNames.get(title);

    return <Card
        slotContent={<Flex justify="center"><h4 style={{textAlign: 'center', color: "white"}}>{AppNames.get(title)}</h4></Flex>}
        slotMedia={iconName ?
            <Image width="200" height="150" src={`/images/${iconName}`} alt={title} >
            </Image> :
            <Image height="150" width="150" src={`/images/holoscan.png`} alt={title} ></Image>
        }
        onClick={() => handleClick(title)}
        id={title}
        css={{margin: 20, padding: "30px, 20px", width: 300, height: 250, color: "#76b900", cursor: 'pointer'}}
    />
}
